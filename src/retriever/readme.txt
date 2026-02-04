Before running the teacher model, which is in main.py, first run main_pretrain_chose.py

main_pretrain_chose.py will do two thing. 

1. pretrain a topic entity choosen model. Given a question, it will choose the topic entity from the question. 
Remember that for conversation QA, everytime when user asks a new question, we need to decide the new topic entity is the global topic entity or the answer of last question. 
main_pretrain_chose.py is a pretrained classifier which does this. 

After we train the main_pretrain_chose.py, we find the model always choose the global topic entity as the new topic entity, so during the later training, 
for both the teacher model and student model, we always set the new topic entity as global topic entity for all questions in a conversation. 

2. main_pretrain_chose.py will also train an lstm model to warm up the teacher model.




when running the model, no parameters are required in the command line.
Just change the correspoinding config_xx.py