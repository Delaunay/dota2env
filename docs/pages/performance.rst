Performance
===========

* Observation Time: 133.33
* Reading action file ``IPC_recv.lua`` & take action: ``2ms-5ms``
* Process State deltas: ``20-50 ms``
* Time remaining for inference = ``133 - 5 - 50 = 78 ms``


To improve performance main work should be focusing on speeding up state parsing.

Ideas:
* Move the stitching to the state process so more work is done in parallel


Inference
~~~~~~~~~

Inference will be parallel enough when using pytorch, the model will compute actions for all 10 bots actions
at the same time (essentially batch = 10)
