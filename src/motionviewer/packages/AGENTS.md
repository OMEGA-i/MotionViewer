# Package module

- `motionviewer.packages` is the public interface for Sample Packing Protocol consumers.
- Treat package contents as untrusted input. Reject traversal, ambiguous roots, missing members, and invalid metadata.
- Directory and tar adapters must expose the same `PackageStore` semantics.
- Selection and planning stay pure; materialization and copying stay in the store/interface implementation.
- A wire-format change requires package parsing tests.
