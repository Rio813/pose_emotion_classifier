import predictions.classPred

pred = predictions.classPred.Pred('epoch395_acc81.25_model.pth.tar', data_path='features.h5')
label = pred.generate_predictions(pred.generate_data())
print(label)
