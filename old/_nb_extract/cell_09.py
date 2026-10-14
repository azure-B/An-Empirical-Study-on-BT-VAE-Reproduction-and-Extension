def make_seg_loaders(data, batch_size):
    N = len(data); N_train = int(0.9*N); 
    N_dev = int((N - N_train)); 
    print(N_train)
    print(N_dev)
    train_data, dev_data = D.random_split(
        data, [N_train, N_dev])

    train_loader = D.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
 
    dev_loader = D.DataLoader(
        dev_data, batch_size=batch_size, shuffle=True)
    print(len(train_loader))
    print(len(dev_loader))      
    return train_loader, dev_loader