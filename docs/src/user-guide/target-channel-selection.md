# Target channel selection design

Issue [#440](https://github.com/alan-turing-institute/icenet-mp/issues/440) identifies a limitation in the current DDPM processor. Target latent channels are represented by a single offset plus the target channel count, which assumes that every target channel occupies one contiguous slice of the combined latent tensor.

This page proposes a representation that removes that assumption while keeping the current configuration path compatible.

## Requirements

The target-channel representation should:

- preserve the current contiguous-slice behaviour without changing existing configs;
- represent target channels from more than one non-adjacent encoder;
- preserve the order that maps selected combined-latent channels to target-latent channels;
- keep channel-selection logic outside the DDPM sampling algorithm;
- provide a path for future fusion layers where target values cannot be represented by simple indexing.

## Proposed abstraction

Represent the relationship between the combined latent space and the target latent space as a small mapping object rather than storing slice bounds in `DDPMProcessor`.

```python
from typing import Protocol


class TargetLatentMapping(Protocol):
    def extract(self, combined): ...
    def insert(self, combined, target): ...
```

The processor uses only the two operations it actually needs:

1. `extract(combined)` returns target channels in target-latent order.
2. `insert(combined, target)` returns a combined-latent tensor with the target values written back into the appropriate representation.

The first implementation should be index-based.

```python
@dataclass(frozen=True)
class IndexTargetMapping:
    groups: tuple[tuple[int, ...], ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(index for group in self.groups for index in group)
```

The group structure preserves source boundaries for diagnostics and configuration while `indices` provides the ordered flattened selection needed for tensor indexing.

Examples:

```python
# Current behaviour, target occupies channels 4, 5, 6
IndexTargetMapping(groups=((4, 5, 6),))

# Two target groups from non-adjacent encoders
IndexTargetMapping(groups=((1, 2), (7, 9)))
```

## Backwards compatibility

The existing `target_channel_offset` path can be translated when the model is built:

```python
indices = tuple(range(target_channel_offset, target_channel_offset + c_target))
mapping = IndexTargetMapping(groups=(indices,))
```

Existing configs therefore continue to describe the current concatenated-latent architecture. `target_channel_offset` can remain supported while model construction migrates toward an explicit mapping.

`DDPMProcessor` should then depend on the mapping for insertion and extraction instead of calculating `target_slice_start` and `target_slice_end` itself.

## Validation

An index-based mapping should reject configurations where:

- an index is negative or outside the combined latent channel range;
- the same combined-latent index appears more than once;
- the flattened number of selected channels does not equal the target latent channel count;
- group ordering is ambiguous or empty groups are supplied.

Tests should cover the current contiguous case, a non-zero contiguous offset, multiple non-contiguous groups, invalid indices, duplicate indices, and preservation of non-target channels during DDPM rollout.

## Learned fusion

A learned fusion layer can mix information across every combined-latent channel, so no index list can in general recover a target slice. The mapping interface keeps that case separate from the DDPM algorithm.

A later `ProjectedTargetMapping` can implement `extract` and `insert` using learned or fixed projection modules. The processor API can stay unchanged because it operates through `TargetLatentMapping` rather than assuming a concatenated representation.

## Migration path

The implementation can be introduced incrementally:

1. Add `TargetLatentMapping` and `IndexTargetMapping` with validation and unit tests.
2. Construct an `IndexTargetMapping` from the existing `target_channel_offset` in `EncodeProcessDecode` and `ProcessorStage`.
3. Replace direct DDPM target-slice assignment with mapping `insert` and, where needed, `extract`.
4. Keep the offset-based constructor path during a deprecation period.
5. Add projected mappings only when a non-concatenation fusion layer requires them.

This keeps the current model behaviour stable while making the target-channel contract explicit enough for multi-source and future fused latent representations.
