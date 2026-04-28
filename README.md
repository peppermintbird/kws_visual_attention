# KWS visual attention



```bash
kws_visual_attention/
├── run.py
├── data/
│   ├── __init__.py
│   ├── spiking_dataset.py      ← SpikingDS (one sample → graph)
│   └── spiking_commands.py     ← SpikingCommands (LightningDataModule)
├── keyword_spotter/
│   ├── __init__.py
│   ├── kws_spotter.py
│   └── ... (your GCNN scripts: model, edge_generator, config, etc.)
├── keyword_modulator/
│   ├── __init__.py
│   ├── helpers_kws_mod.py
│   └── kws_mod.py
└── visual_attention/
    ├── __init__.py
    ├── helpers_visual_att.py
    ├── visual_attention.py
    └── video.py
```

## Keyword-spotter 
**Input:** raw audio (converted to events) <br>
**Output:** confidence, keyword class

### Components:
1. Output from GCNN
```python
output  : Tensor [1, T]           # sigmoid confidence over time
cls     : Tensor [1, num_cls, T]  # softmax class scores over time
```

2. Reduced to output peak confidence only
```python
t_max       = argmax(output[0])          # time step of highest confidence
confidence  = output[0, t_max]           # scalar ∈ [0, 1]
class_id    = argmax(cls[0, :, t_max])   # int, e.g. 0=left 1=right
```
## Keyword-modulator
Connected GCNN spotter output to keyword-modulatior pipeline. 
<br>
**Input:** confidence, keyword class <br>
**Output:**
### Components:
threshold gate    fired = confidence >= T  →  W = confidence
boost             B = W × alpha

## Visual attention
**Input:** <br>
**Output:** 
### Components:
neighbor lookup   target = NEIGHBOR[current_quad][keyword]
saliency update   saliency[target] = B   (others unchanged)
normalization     norm[i] = saliency[i] / Σ saliency

# kws_visual_attention
