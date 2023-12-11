from load_data import train_loader, val_loader, test_loader, word_vocab, gloss_vocab

vocab_size = len(word_vocab)
output_size = len(gloss_vocab)

n_layers = 2

student_embed_size = 64
student_hidden_size = 128

teacher_embed_size = 128
teacher_hidden_size = 256

rule_penalty = 1
loss_balancer = 1