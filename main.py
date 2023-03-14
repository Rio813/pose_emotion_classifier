import predictions.classPred

pred = predictions.classPred.Pred(model_path='epoch395_acc81.25_model.pth.tar', video_name="demo.mp4")

label = pred.generate_predictions(pred.generate_data())

print(label)
