<div align="center">
  <img src="./img/logo_grid.png" alt="Logo" width="200">
</div>

# SWIFT-AI: An Extremely Fast System For Gigapixel Visual Understanding In Science

<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/liwenxi/SWIFT-AI?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/liwenxi/SWIFT-AI?color=red" alt="Issues">
<img src="https://img.shields.io/badge/python-3.8-purple.svg" alt="Python">
<!-- **Authors:** -->
<!-- **_¹  [Wenxi Li](https://liwenxi.github.io/)_** -->

<!-- **Affiliations:** -->

<!-- _¹ Shanghai Jiao Tong University_ -->

</div>

Welcome to the dawn of a new era in scientific research with SWIFT AI, our ground-breaking system that harnesses the power of deep learning and gigapixel imagery to revolutionize visual understanding across diverse scientific fields. Pioneering in speed and accuracy, SWIFT AI promises to turn minutes into seconds, offering a giant leap in efficiency and accuracy, thereby empowering researchers and propelling the boundaries of knowledge and discovery.



#### 📰 <a href="https://xxx" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Paper</a>     :building_construction: <a href="https:/xxx" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Model (via Google)</a>    :building_construction: <a href="https://pan.baidu.com/s/1j2WMkmEj0nqOOctiQGj2Wg?pwd=v7mi" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Model (via Baidu)</a>    :card_file_box: <a href="https://www.gigavision.cn/data/news?nav=DataSet%20Panda&type=nav&t=1689145968317" style="color: black; text-decoration: underline;text-decoration-style: dotted;">Dataset</a>    :bricks: [Code](#usage)    :monocle_face: Video    :technologist: Demo    



## Table of Contents 📚

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work and Contributions](#future-work-and-contributions)



## Key Features 🔑

SWIFT-AI will become the third eye of researchers, helping to observe objects in a large field of view, and assisting the discovery of strong gravitational lenses by the <a href="https://www.lsst.org/science/transient-optical-sky">LSST project</a>.

![Zoom into NGC 1333](img/Galaxy.gif)

## Gigapixel-level Datasets 

### PANDA
- [PANDA] A Gigapixel-level Human-centric Video Dataset [[Link](https://gigavision.cn/data/news/?nav=DataSet%20Panda&type=nav%2Findex.html)]

PANDA is the first gigaPixel-level humAN-centric viDeo dAtaset, for large-scale, long-term, and multi-object visual analysis. The videos in PANDA were captured by a gigapixel camera and cover real-world large-scale scenes with both wide field-of-view  (~1km² area) and high resolution details (~gigapixel-level/frame). The scenes may contain 4k head counts with over 100× scale variation. PANDA provides enriched and hierarchical ground-truth annotations, including 15,974.6k bounding boxes, 111.8k fine-grained attribute labels, 12.7k trajectories, 2.2k groups and 2.9k interactions.

### Camelyon
- [Camelyon] 1399 H&E-stained sentinel lymph node sections of breast cancer patients: the CAMELYON dataset [[Link](https://gigadb.org/dataset/100439)]

The goal of this challenge is to evaluate new and existing algorithms for automated detection of metastases in hematoxylin and eosin (H&E) stained whole-slide images of lymph node sections. This task has a high clinical relevance but requires large amounts of reading time from pathologists. Therefore, a successful solution would hold great promise to reduce the workload of the pathologists while at the same time reduce the subjectivity in diagnosis. This will be the first challenge using whole-slide images in histopathology. The challenge will run for two years. The 2016 challenge will focus on sentinel lymph nodes of breast cancer patients and will provide a large dataset from both the Radboud University Medical Center (Nijmegen, the Netherlands), as well as the University Medical Center Utrecht (Utrecht, the Netherlands).

### Galaxy
- [Galaxy] Real galaxy dataset extracted from the HST COSMOS survey for use with GalSim. [[Link](https://zenodo.org/records/3242143)]


### * We have open-sourced the complete code for InhibitionFormer, a framework built on [MMDetection](https://github.com/open-mmlab/mmdetection). Additionally, you can create and integrate your own dataset by following the provided tutorials.
