# Packaging validation

- All supplied Python sources parse; all JSON files parse.
- Six synthetic test suites pass: additional analyses, robustness, background transfer, DINOv2 ViT-B, MegaDescriptor-L and human-view analysis.
- Tests run without EQUINE_DATA_ROOT, confirming the environment variable is no longer required at import time.
- Both 10,000-replicate paired identity bootstrap analyses rerun from the supplied predictions. Point estimates and interval bounds match the recorded results exactly.
- Source/data manifests and numeric scientific outputs are retained. Package paths and documentation are sanitized; this does not certify anonymity of external services or upstream provenance.
- Actual deep-model extraction, segmentation and training are not rerun during packaging: PyTorch and corpus images are unavailable here. Synthetic extraction tests simulate model backends.
- Original runtime result folders are renamed with recorded_ prefixes. JSON-only segmentation caches are removed because their binary counterparts are absent.
- The root README is new and in English. Original module documentation and comments may remain in French.
