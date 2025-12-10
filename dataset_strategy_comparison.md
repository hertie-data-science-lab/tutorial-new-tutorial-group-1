# Dataset Strategy Comparison for Few-Shot Rooftop Segmentation Tutorial

**Goal**: Compare three different dataset strategies for teaching few-shot learning for rooftop/building segmentation.

#### ATTENTION: This document is AI generated, and should be used as a checklist / orientative document. Information have been checked, and should be acceptably accurate.

---

## Executive Summary

| Strategy | Complexity | Realism | Tutorial Clarity | Implementation Effort | Recommendation |
|----------|------------|---------|------------------|----------------------|----------------|
| **Geneva + Inria** | Medium | High | ⭐⭐⭐⭐⭐ | Medium | **Best for education** |
| **Only Inria** | Low-Medium | Medium-High | ⭐⭐⭐⭐ | Low | **Best for simplicity** |
| **Only RID** | High | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | High | **Best for realism** |

**Quick Recommendation**:
- **For teaching fundamentals**: Use Geneva + Inria (Strategy 1)
- **For quick implementation**: Use Only Inria (Strategy 2)
- **For real-world impact**: Use Only RID (Strategy 3)

---

## Dataset Deep Dive

### Geneva Satellite Dataset
**Source**: HuggingFace (`raphaelattias/overfitteam-geneva-satellite-images`)

**Key Facts**:
- **Size**: 1,050 labeled image-mask pairs
- **Task**: Binary segmentation (rooftop vs background)
- **Geographic splits**: 3 grids (1301_11, 1301_13, 1301_31)
- **Image size**: 250x250 pixels
- **Categories**: All, Industrial, Residential
- **Characteristics**: Single city, well-defined geographic regions

**Strengths**:
- ✅ Clean binary task
- ✅ Built-in geographic splits
- ✅ Easy to use (already on HuggingFace)
- ✅ Good size for tutorial (not too large)
- ✅ Clear within-city domain shift

**Limitations**:
- ❌ Only one city
- ❌ Relatively small compared to others
- ❌ Simple task (binary only)

---

### Inria Aerial Image Dataset
**Source**: https://project.inria.fr/aerialimagelabeling/
**HuggingFace**: `Jonathan/INRIA-Aerial-Dataset`

**Key Facts**:
- **Size**: 360 images (180 train + 180 test)
- **Coverage**: 180 km² across 5 cities
- **Cities**:
  - **Austin, Texas** (US) - Suburban sprawl
  - **Chicago, Illinois** (US) - Dense urban
  - **Kitsap County, Washington** (US) - Rural/suburban
  - **Vienna, Austria** (EU) - European urban
  - **Tyrol-Innsbruck, Austria** (EU) - Mountain town
- **Task**: Binary segmentation (building vs background)
- **Image size**: 5000x5000 pixels (need tiling)
- **Resolution**: 0.3m per pixel

**Strengths**:
- ✅ Multiple cities (5 different domains)
- ✅ Geographic diversity (US + Europe)
- ✅ Urban diversity (dense city, suburban, rural)
- ✅ Large coverage area
- ✅ Same task as Geneva (binary)
- ✅ Well-documented benchmark dataset
- ✅ Freely available

**Limitations**:
- ❌ Large file sizes
- ❌ Only building footprints (not roof-specific)
- ❌ Fixed train/test split per city

**City Characteristics**:
```
Austin:     Large, low-density sprawl, similar-looking houses
Chicago:    Dense urban, varied building types, tall buildings
Kitsap:     Rural/suburban, scattered buildings, trees
Vienna:     European architecture, moderate density, distinct style
Tyrol:      Mountain town, alpine architecture, complex terrain
```

---

### RID - Roof Information Dataset
**Source**: mediaTUM (TUM) + GitHub (https://github.com/TUMFTM/RID)

**Key Facts**:
- **Size**: 1,880 images (+ 26 for annotation experiment)
- **Task 1**: Roof segments - 18 classes
  - Background + 16 azimuth directions (N, NNE, NE, ENE, E, ...) + Flat
- **Task 2**: Roof superstructures - Multiple classes
  - PV modules, windows, chimneys, dormer windows, etc.
- **Image source**: Google Maps Static API (georeferenced)
- **Cities**: Multiple (not explicitly listed, but multi-city dataset)
- **Split**: 1880 images with train/val/test splits provided

**Strengths**:
- ✅ **Directly relevant**: Designed for solar panel assessment
- ✅ **Multi-class**: More realistic task complexity
- ✅ **Rich annotations**: Both segments and superstructures
- ✅ **Large size**: 1,880 images
- ✅ **Quality control**: Reviewed annotations, multiple labelers
- ✅ **Practical application**: Real-world solar potential assessment
- ✅ **Code available**: GitHub repo with data preparation

**Limitations**:
- ❌ **Complex**: Multi-class segmentation harder to learn
- ❌ **Google Maps imagery**: Different from satellite (closer, angled)

**Task Complexity Comparison**:
```
Geneva:  2 classes  (rooftop, background)
Inria:   2 classes  (building, background)
RID-S:   18 classes (16 directions + flat + background)
RID-SS:  ~10 classes (PV, window, chimney, dormer, ...)
```
---

## Strategy 1: Geneva + Inria

### Overview
Use Geneva as base dataset, Inria cities as cross-city transfer targets.

### Task Structure
```
Phase 1: Within-City Transfer (Geneva)
├─ Train: Grid 1301_11 (295 images)
├─ Test:  Grid 1301_13 (76 images) - Different neighborhood
└─ Test:  Grid 1301_31 (49 images) - Different neighborhood

Phase 2: Cross-City Transfer (Geneva → Inria)
├─ Train: Grid 1301_11 (Geneva)
├─ Test:  Vienna (European, similar to Geneva)
├─ Test:  Austin (US suburban, different architecture)
└─ Optional: Chicago, Kitsap, Tyrol (varying difficulty)
```

### Few-Shot Setup
**Training** (on Geneva Grid 1301_11):
- Fine-tuning baseline: Standard supervised learning
- Prototypical Networks: Episodic training (K=3-5 per episode)
- PANet: Episodic training with MAP

**Evaluation** (on target domains):
```python
For each target (1301_13, 1301_31, Vienna, Austin):
    For K in [1, 3, 5, 10, 20]:
        # Select K support examples from target
        support_set = random_sample(target_data, K)
        query_set = remaining_data

        # Apply method
        predictions = method.predict(support_set, query_set)

        # Evaluate
        iou = compute_iou(predictions, query_masks)
```

### Expected Domain Shift
```
Geneva 1301_11 → Geneva 1301_13:  SMALL (same city, diff neighborhood)
Geneva 1301_11 → Geneva 1301_31:  SMALL-MEDIUM (different area)
Geneva 1301_11 → Vienna:          MEDIUM (Euro→Euro, diff city)
Geneva 1301_11 → Austin:          LARGE (Euro→US, diff architecture)
```

### Advantages
✅ **Progressive difficulty**: Small → Large domain shift
✅ **Same task throughout**: Binary segmentation
✅ **Clear learning progression**: Students see increasing challenge
✅ **Multiple test cases**: 5 target domains (2 Geneva + 3 Inria)
✅ **Well-documented**: Both datasets established
✅ **Best for teaching**: Clear concept progression

### Disadvantages
❌ **Two datasets**: More setup complexity
❌ **Preprocessing needed**: Tile Inria 5000x5000 → 250x250
❌ **Different sources**: Satellite vs aerial imagery mix

### Implementation Complexity: **Medium** (6/10)

### Tutorial Timeline
```
Week 1: Geneva setup + baseline (Grid 1301_11 → 1301_13/31)
Week 2: Prototypical Networks on Geneva
Week 3: PANet on Geneva + comparison
Week 4: Cross-city transfer (Vienna, Austin)
```

### Key Learning Outcomes
1. Understand few-shot learning fundamentals
2. See domain adaptation on same task
3. Compare methods across varying domain shifts
4. Learn practical deployment scenario (city → city)

---

## Strategy 2: Only Inria (Multi-City)

### Overview
Use only Inria dataset, train on one city, test on others.

### Task Structure
```
Single-Dataset Multi-City Transfer

Train: Vienna (36 train images × tiled = ~576 patches)

Test Domains:
├─ Austin   (US suburban - LARGE shift from Vienna)
├─ Chicago  (US urban - LARGE shift, but urban like Vienna)
├─ Kitsap   (US rural - VERY LARGE shift)
└─ Tyrol    (Austrian mountain - MEDIUM shift, both European)
```

### Alternative Training Cities
```
Option A: Train on Vienna (European urban)
  → Test on Austin, Chicago, Kitsap, Tyrol
  → Bias: European → American

Option B: Train on Austin (US suburban)
  → Test on Vienna, Chicago, Kitsap, Tyrol
  → Bias: Suburban → diverse targets

Option C: Train on Chicago (Dense urban)
  → Test on Vienna, Austin, Kitsap, Tyrol
  → Bias: Dense urban → varying densities
```

### Few-Shot Setup
**Training** (e.g., on Vienna):
```python
# Tile Vienna images into 250x250 patches
vienna_patches = tile_images(vienna_train, patch_size=250)
# Result: ~576 training patches

# Episodic training
for episode in range(num_episodes):
    support, query = sample_episode(vienna_patches, K=5, Q=5)
    # Train meta-learning model
```

**Evaluation** (on other cities):
```python
For each target_city in [Austin, Chicago, Kitsap, Tyrol]:
    # Tile target city images
    city_patches = tile_images(target_city, patch_size=250)

    For K in [1, 3, 5, 10, 20]:
        support = random_sample(city_patches, K)
        query = remaining_patches

        predictions = method.predict(support, query)
        iou = compute_iou(predictions, query_masks)
```

### Expected Domain Shift
```
Vienna → Tyrol:    MEDIUM (both Austrian, diff terrain)
Vienna → Chicago:  LARGE (Euro→US, but both urban)
Vienna → Austin:   VERY LARGE (Euro→US, urban→suburban)
Vienna → Kitsap:   VERY LARGE (Euro→US, urban→rural)
```

### Advantages
✅ **Single dataset**: Simpler setup, one data source
✅ **Multiple targets**: 4 different cities to test on
✅ **Established benchmark**: Well-known dataset
✅ **Same task**: Binary segmentation throughout
✅ **Good size**: After tiling, ~500-600 patches per city
✅ **Geographic diversity**: US + Europe, urban + suburban + rural
✅ **Easy to extend**: Can add all 5 cities as targets

### Disadvantages
❌ **No within-city baseline**: Can't show small domain shift first
❌ **All shifts are large**: Harder to see method differences
❌ **Requires tiling**: Must preprocess 5000x5000 images
❌ **Building vs rooftop**: Not exactly rooftops (full building footprints)
❌ **Less progression**: Jumps straight to hard cross-city transfer

### Implementation Complexity: **Low-Medium** (4/10)

### Tutorial Timeline
```
Week 1: Inria setup + tiling + Vienna baseline
Week 2: Prototypical Networks on Vienna
Week 3: PANet on Vienna + comparison
Week 4: Multi-city evaluation (4 cities)
```

### Key Learning Outcomes
1. Understand few-shot learning on real cross-city task
2. See performance across very different domains
3. Compare method robustness to large domain shifts
4. Learn practical multi-city deployment

### Recommendation for This Strategy
**Best training city**: Vienna
- **Why**: European city, moderate density, good variety
- **Test on**: Austin (easiest), Chicago (medium), Kitsap (hardest)
- **Skip**: Tyrol initially (too similar to Vienna)

---

## Strategy 3: Only RID (Multi-Class, Multi-City)

### Overview
Use only RID dataset for realistic solar panel assessment task.

### Task Structure

**Option A: Roof Segments (18-class)**
```
Task: Predict roof orientation (azimuth + flat)
Classes: Background, N, NNE, NE, ENE, E, ESE, SE, SSE,
         S, SSW, SW, WSW, W, WNW, NW, NNW, Flat

Train: Subset of cities (e.g., 60% of 1880 = ~1128 images)
Val:   20% (~376 images)
Test:  20% (~376 images)

Few-shot: Use K examples from test set cities
```

**Option B: Roof Superstructures (Multi-class)**
```
Task: Detect roof features for solar assessment
Classes: Background, PV module, Window, Chimney,
         Dormer, Satellite dish, etc.

Same split as Option A
```

**Option C: Hierarchical (Recommended)**
```
Phase 1: Coarse segmentation (Roof vs Background)
  - Simplify RID masks to binary
  - Train baseline model

Phase 2: Fine-grained segmentation (Superstructures)
  - Given roof regions, detect PV modules, windows, etc.
  - Use K examples for few-shot superstructure detection

This shows: Can we adapt from "find roofs" to "analyze roofs"?
```

### Few-Shot Setup (Option B - Superstructures)

**Data Preparation**:
```python
# RID provides georeferenced images + annotations
# Use their GitHub code to generate masks

from rid_tools import generate_masks

# Generate superstructure masks
masks = generate_masks(
    annotations_path='RID/annotations/',
    task='superstructures',
    classes=['background', 'pvmodule', 'window', 'chimney', 'dormer']
)
```

**Training**:
```python
# Option 1: Standard split (not city-based)
train_images = rid_data[:1128]  # 60%
val_images = rid_data[1128:1504]  # 20%
test_images = rid_data[1504:]  # 20%

# Option 2: City-based split (if city info available)
train_cities = ['CityA', 'CityB', 'CityC']
test_cities = ['CityD', 'CityE']

# Episodic training on train set
for episode in range(num_episodes):
    # N-way K-shot episodes
    classes = sample_classes(N=3)  # e.g., pvmodule, window, chimney
    support, query = sample_episode(classes, K=5, Q=5)
```

**Evaluation**:
```python
# Few-shot evaluation on test set
For K in [1, 3, 5, 10, 20]:
    # Sample K examples of each class
    support_set = sample_per_class(test_images, K_per_class=K)
    query_set = remaining_test_images

    # Predict
    predictions = method.predict(support_set, query_set)

    # Compute metrics
    iou_per_class = compute_iou(predictions, query_masks, classes)
    mean_iou = iou_per_class.mean()
```

### Expected Challenges
```
Binary (Geneva/Inria):  Easy - 2 classes, clear boundaries
RID Segments:           Hard - 18 classes, subtle differences (NNE vs NE)
RID Superstructures:    Medium-Hard - Fewer classes but small objects
```

### Advantages
✅ **Most realistic**: Actual solar panel assessment task
✅ **Rich annotations**: Multiple semantic levels
✅ **Direct application**: PV module detection is the end goal
✅ **Multi-class few-shot**: Shows method capability on hard task
✅ **Single dataset**: No need to integrate multiple sources
✅ **Large size**: 1,880 images is substantial
✅ **Quality data**: Reviewed annotations, multiple labelers
✅ **Code available**: GitHub repo with utilities

### Disadvantages
❌ **High complexity**: Multi-class harder to learn/teach
❌ **Requires preprocessing**: Must generate masks from annotations
❌ **Different task**: Not comparable to Geneva/Inria
❌ **Harder baselines**: Multi-class needs more careful implementation
❌ **Less established**: Newer dataset, fewer examples to follow
❌ **License restrictions**: CC-BY-NC (non-commercial)
❌ **Google Maps imagery**: Different from satellite
❌ **Small objects**: PV modules, windows are small, harder to segment

### Implementation Complexity: **High** (8/10)

### Tutorial Timeline
```
Week 1: RID setup + mask generation + data exploration
Week 2: Binary baseline + multi-class baseline
Week 3-4: Prototypical Networks (need more time for multi-class)
Week 5: PANet + comparison
Week 6: Analysis and visualization
```

### Key Learning Outcomes
1. Understand few-shot learning on complex multi-class task
2. Learn N-way K-shot episodic training
3. See real-world application (solar assessment)
4. Handle class imbalance (some classes rare)
5. Deal with small objects (PV modules)

### Recommendation for This Strategy
**If using RID, focus on**:
- **Task**: Roof superstructures (more interpretable than 18 directions)
- **Classes**: PV module, Window, Chimney, Background (4-way task)
- **Approach**: Start with binary (roof vs background), then few-shot superstructures
- **K values**: Use higher K (5-20) due to multi-class complexity

---

## Head-to-Head Comparison

### 1. Tutorial Clarity

**Winner: Geneva + Inria** ⭐⭐⭐⭐⭐

**Reasoning**:
- Clear progression: small shift (Geneva grids) → large shift (Inria cities)
- Same task (binary) throughout makes concept learning easier
- Students can focus on few-shot methods, not task complexity

**Rankings**:
1. Geneva + Inria (5/5) - Clearest learning path
2. Only Inria (4/5) - Still clear, but no easy baseline
3. Only RID (3/5) - Multi-class adds complexity

---

### 2. Real-World Relevance

**Winner: Only RID** ⭐⭐⭐⭐⭐

**Reasoning**:
- Direct solar panel application (PV module detection)
- Multi-class is more realistic than binary
- Actually detects roof features, not just roofs

**Rankings**:
1. Only RID (5/5) - Most realistic solar assessment
2. Geneva + Inria (4/5) - City deployment scenario realistic
3. Only Inria (3.5/5) - Building footprints less specific than roofs

---

### 3. Implementation Effort

**Winner: Only Inria** ⭐⭐⭐⭐⭐

**Reasoning**:
- Single dataset to manage
- Simple binary task
- Well-documented, many examples

**Rankings**:
1. Only Inria (5/5) - Simplest
2. Geneva + Inria (3/5) - Two datasets, tiling needed
3. Only RID (2/5) - Multi-class, preprocessing, less docs

---

### 4. Dataset Size & Diversity

**Winner: Only RID** (size) / **Only Inria** (diversity)

**For Size**:
1. Only RID: 1,880 images
2. Geneva + Inria: ~1,050 + ~360 (after tiling ~1,050 + 1,800) = ~2,850
3. Only Inria: ~360 (after tiling ~1,800)

**For Geographic Diversity**:
1. Only Inria: 5 cities, 3 countries, varied terrain
2. Geneva + Inria: 6 cities total
3. Only RID: Multiple cities (but not specified)

