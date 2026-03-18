Blob Data Schema
================

Flowcept stores binary payload metadata in the ``objects`` collection.
This document describes the logical schema represented by ``BlobObject``.

Required Fields
---------------

- **object_id** (str):
  Unique identifier for the blob record.

- **version** (int):
  Object version number. starts at ``0`` and increments by ``1`` on each update of the same ``object_id``.

Optional Fields
---------------

- **task_id** (str):
  Task linkage field. Use this when the blob was produced or consumed by a specific task.

- **workflow_id** (str):
  Workflow linkage field. Use this to associate the blob with a workflow execution.
  If ``workflow_id`` is not explicitly passed when saving, Flowcept uses
  ``Flowcept.current_workflow_id`` when available.

- **type** (str):
  User-defined category label for the blob.
  Typical values: ``ml_model`` (trained model/checkpoint bytes), ``dataset_snapshot`` (frozen dataset payloads),
  ``artifact`` (generic serialized outputs), ``input_file`` (uploaded/source binary inputs),
  and ``embedding_index`` (vector index payloads).

- **custom_metadata** (dict):
  Free-form dictionary for additional tags and attributes (for example, ``{"framework": "torch", "stage": "best"}``).

- **created_at** (datetime, UTC):
  Logical object creation timestamp.

- **created_by** (str):
  Logical object creator identifier.

- **updated_at** (datetime, UTC):
  Latest update timestamp.

- **updated_by** (str):
  Latest updater identifier.

- **prev_version** (int or null):
  Previous latest version number (``None`` for first insert in controlled mode).

- **object_size_bytes** (int):
  Payload size in bytes when available.

- **data_sha256** (str):
  SHA-256 hash of payload bytes for fast equality checks and integrity verification.

- **data_hash_algo** (str):
  Hash algorithm label for the payload fingerprint (currently ``sha256``).

Notes
-----

- Binary payload bytes are stored either **in-object** (``data`` field) or out-of-line in GridFS depending on ``save_data_in_collection``.
- When storage mode is GridFS, the document keeps ``grid_fs_file_id`` as the pointer to payload bytes.
- ``BlobObject`` captures metadata/linkage fields; payload storage location is implementation-specific.
- In version-control mode, Flowcept keeps latest in ``objects`` and append-only older versions in ``object_history``.
