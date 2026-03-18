Versioned Model Example
=======================

This example mirrors the ``single_layer_perceptron_test`` flow and focuses on versioned
checkpoint storage for the best validation-loss model.

Purpose
-------

- Show Git-like version control with ``control_version=True`` for model checkpoints.
- Show two alternative save paths:
  - generic ``ml_model`` path (works for any model payload you serialize),
  - PyTorch-specific helper path (works only for PyTorch models).
- Show simple inference after loading the best saved model from each path.

Use one save path in real projects. This example uses both only to demonstrate and test both APIs.

Training + Checkpoint Save (Two Paths)
--------------------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from flowcept import Flowcept
   from flowcept.instrumentation.flowcept_task import get_current_context_task_id

   class SingleLayerPerceptron(nn.Module):
       def __init__(self, input_size):
           super().__init__()
           self.layer = nn.Linear(input_size, 1)

       def forward(self, x):
           return torch.sigmoid(self.layer(x))

   def validate(model, criterion, x_val, y_val):
       model.eval()
       with torch.no_grad():
           outputs = model(x_val)
           loss = criterion(outputs, y_val)
       return loss.item()

   def train_and_checkpoint(x_train, y_train, x_val, y_val, epochs=4, checkpoint_check=2):
       model = SingleLayerPerceptron(input_size=2)
       criterion = nn.BCELoss()
       optimizer = optim.SGD(model.parameters(), lr=0.1)

       best_val_loss = float("inf")
       ml_model_object_id = None
       torch_model_object_id = None
       current_task_id = get_current_context_task_id()

       for epoch in range(1, epochs + 1):
           model.train()
           outputs = model(x_train)
           loss = criterion(outputs, y_train)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           current_val_loss = validate(model, criterion, x_val, y_val)
           if epoch % checkpoint_check == 0 and current_val_loss < best_val_loss:
               best_val_loss = current_val_loss
               metadata = {"loss": best_val_loss, "checkpoint_epoch": epoch}

               # Path A: generic object API ("ml_model"), version-controlled.
               ml_model_object_id = Flowcept.db.save_or_update_ml_model(
                   object=model.state_dict(),
                   object_id=ml_model_object_id,
                   task_id=current_task_id,
                   custom_metadata=metadata,
                   save_data_in_collection=True,
                   pickle=True,
                   control_version=True,
               )

               # Path B: PyTorch helper API, also version-controlled.
               torch_model_object_id = Flowcept.db.save_or_update_torch_model(
                   model=model,
                   object_id=torch_model_object_id,
                   task_id=current_task_id,
                   custom_metadata=metadata,
                   control_version=True,
               )

       return ml_model_object_id, torch_model_object_id

Load Best Model + Inference
---------------------------

.. code-block:: python

   import pickle
   import torch
   from flowcept import Flowcept

   sample = torch.randn(1, 2)

   # Load best model from Path A (generic ml_model with pickled state_dict payload).
   best_ml_model_blob = Flowcept.db.get_ml_model(ml_model_object_id)
   ml_state_dict = pickle.loads(best_ml_model_blob.data)
   ml_model = SingleLayerPerceptron(input_size=2)
   ml_model.load_state_dict(ml_state_dict)
   ml_model.eval()
   with torch.no_grad():
       ml_pred = ml_model(sample)

   # Load best model from Path B (PyTorch helper).
   reloaded_torch_model = SingleLayerPerceptron(input_size=2)
   Flowcept.db.load_torch_model(reloaded_torch_model, torch_model_object_id)
   reloaded_torch_model.eval()
   with torch.no_grad():
       torch_pred = reloaded_torch_model(sample)

   assert ml_pred.shape == (1, 1)
   assert torch_pred.shape == (1, 1)

Why Two Paths?
--------------

- ``save_or_update_ml_model`` is format-agnostic and can store serialized payloads from non-PyTorch models too.
- ``save_or_update_torch_model`` is convenience tooling for PyTorch ``state_dict`` save/load.

Both support version control when ``control_version=True`` and ``object_id`` is reused across updates.
