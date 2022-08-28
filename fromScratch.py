# CSCI-P556 Final Project, Ethan Eldridge @etmeldr
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator 
from keras import Model 
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import os
import sys
import shutil

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def fromScratch(EPOCHS, LR):
    CONFIG = "E-"+str(EPOCHS)+"_LR-"+str(LR)

    print("\n\n")
    print("***STAGE 1 INITIALIZED***")
    print("\n\n")

    # Training Stage 1: PlantVillage DataSet
    """
    Stage 1 of the project trains the VGG-16 model from scratch on the entire PlantVillage dataset, 
    minus the Healthy and Northern Leaf Blight images of corn. These sets were excluded in Stage 1 
    as they will be used as data in Stage 2. 
    """

    TRAIN = 0.7
    TEST = 0.1
    VALID = 1-(TRAIN+TEST)

    # plotImages function taken from Tensorflow website tutorial
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    print("\n\n")
    print("***STAGE 2 INITIALIZED***")
    print("\n\n")



    # Training Stage 2: PlantVillage Dataset
    """
    Stage 2 of the project takes the weights from the Stage 1 model and applies
    transfer learning to classify images of healthy corn and corn with Northern 
    Leaf Blight. 
    """

    #Import and Organize OSF data
    classes = ["blight", "healthy"]

    # Creating file structure
    os.chdir("data_s2")
    if os.path.isdir("train/blight") is False:
        for i in range(len(classes)):
            os.makedirs("train/"+classes[i])
            os.makedirs("test/"+classes[i])
            os.makedirs("valid/"+classes[i])
            pics = [f for f in os.listdir("/Users/ethaneldridge/Documents/CSCI-P556/final_project/data_s2/"+classes[i])]
            trainStop = int(len(pics)*TRAIN)
            testStop = int(len(pics)*(TEST+TRAIN))
            tr = [j for j in range(trainStop)]
            te = [j for j in range(trainStop, testStop)]
            va = [j for j in range(testStop, len(pics))]
            for t in tr:
                shutil.move(classes[i]+"/"+pics[t], "train/"+classes[i])
            for t in te:
                shutil.move(classes[i]+"/"+pics[t], "test/"+classes[i])
            for v in va: 
                shutil.move(classes[i]+"/"+pics[v], "valid/"+classes[i])

    os.chdir("../")
    print(os.getcwd())

    #TODO: Pre-process data
    bSize = 10
    train_path = "data_s2/train"
    test_path = "data_s2/test"
    v_path = "data_s2/valid"

    #TODO: change "_batches" to "_gen"
    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=classes, batch_size=bSize)
    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=classes, batch_size=bSize, shuffle=False)
    valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=v_path, target_size=(224,224), classes=classes, batch_size=bSize)

    images, labels = next(train_batches)

    #TODO: Import model:
    model = VGG16(weights=None, classes=2)
    print(model.summary())

    from tensorflow.keras.optimizers import Adam

    model.compile(optimizer= Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])
    results = model.fit(x=train_batches, validation_data=valid_batches, epochs=EPOCHS, verbose=2)
    model.save("models/model_s2_scratch_"+CONFIG)

    # Visualizing Training and Validation metrics
    print(results.history.keys())

    # summarize history for accuracy
    plt.plot(results.history['accuracy'], label="Training Accuracy")
    plt.plot(results.history['val_accuracy'], label="Validation Accuracy")
    plt.title('Stage 2 Training/Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/acc/model_s2_scratch_acc_"+CONFIG+".png")
    plt.close()
    # summarize history for loss
    plt.plot(results.history['loss'], label="Training Loss")
    plt.plot(results.history['val_loss'], label="Validation Loss")
    plt.title('Stage 2 Training/Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/loss/model_s2_scratch_loss_"+CONFIG+".png")
    plt.close()


    test_images, test_labels = next(test_batches)

    predictions = model.predict(x=test_batches, verbose=0)

    loss, acc = model.evaluate(x=test_batches, batch_size=bSize)

    with open("results/testResults.txt", "a") as file:
        file.write("model_s2_scratch_" + CONFIG+": \n")
        file.write("Mean Test Loss: " + str(loss)+"\n")
        file.write("Mean Test Accuracy: " + str(acc)+"\n\n")
        file.close()
        
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import numpy as np

    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    viz = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    viz.plot()
    plt.savefig("results/cm/model_s2_scratch_cm_"+CONFIG+".png")
    plt.close()

    #TODO: ROC Curve
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_predictions(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plt.savefig("results/roc/model_s2_scratch_ROC_"+CONFIG+".png")
    plt.close()

    #TODO: Precision-Recall Curve
    from sklearn.metrics import PrecisionRecallDisplay

    PrecisionRecallDisplay.from_predictions(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plt.savefig("results/pr/model_s2_scratch_PR_"+CONFIG+".png")
    plt.close()

fromScratch(10, 0.001)