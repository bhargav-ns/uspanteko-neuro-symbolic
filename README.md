**NOTE** : 

The code is functional, but currently, only an extremely messy legacy version of the code with unclean data is available. I am still in the process of cleaning the code and adding the new rule set to this repo.


To train the teacher-student network, install all the packages in requirements.txt.
```
pip install requirements.txt
```

**Switch to the legacy implementation branch by running ```git checkout legacy```.**

Then, navigate to ```neural-LSTM/legacy_implementation/ ``` and simply run ```python main.py```. This will run all the code using the uspanteoko_data.csv file located in the data directory.

The main.py file is super unreadable and messy at the moment. It has data encoding, training, evaluation, and model saving in the same script. This was done for the purpose of submitting a simple batch job on a CURC GPU.

The model is saved to the same directory. By default, it will run a rudimentary parameter search for rule_penalty, train_data_size, penalty_rate, and loss_balancer. To avoid this, hard-code the values at the top and remove the for loops on lines 32-35 in main.py.
