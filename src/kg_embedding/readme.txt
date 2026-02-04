This directory is in use.


It's used to learn the pretrained embedding for entities and relations in the background KG.

training parameters
"args": ["--mode", "train", "--relation_dim", "200", "--hidden_dim", "256", "--gpu", "3", "--freeze", "0", "--batch_size", "128", "--validate_every", "5", "--hops", "1", "--lr", "0.0005", "--entdrop", "0.1", "--reldrop", "0.2",  "--scoredrop", "0.2", "--decay", "1.0", "--model", "ComplEx", "--patience", "5", "--ls", "0.0", "--kg_type", "half"]
