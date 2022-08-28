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

def fromIN(EPOCHS, LR):
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
    #Import and organize data
    classes = ["apple_scab", "apple_rot", "apple_rust", "apple_healthy", "blueberry_healthy", "cherry_healthy", "cherry_mildew", "corn_spot", "corn_rust", "grape_rot", "grape_measles", "grape_healthy", "grape_blight", "orange_greening", "peach_bacteria", "peach_healthy", "pepper_bacteria", "pepper_healthy", "potato_eblight", "potato_healthy", "potato_lblight", "raspberry_healthy", "soybean_healthy", "squash_mildew", "strawberry_healthy", "strawberry_scorch", "tomato_bacteria", "tomato_eblight", "tomato_healthy", "tomato_lblight", "tomato_mold", "tomato_sspot", "tomato_spider", "tomato_tspot", "tomato_mvirus", "tomato_cvirus"]
    print(len(classes))

    TRAIN = 0.7
    TEST = 0.1
    VALID = 1-(TRAIN+TEST)

    # Creating file structure
    os.chdir("data_s1")
    if os.path.isdir("train/apple_scab") is False:
        for i in range(len(classes)):
            os.makedirs("train/"+classes[i])
            os.makedirs("test/"+classes[i])
            os.makedirs("valid/"+classes[i])
            pics = [f for f in os.listdir("/Users/ethaneldridge/Documents/CSCI-P556/final_project/data_s1/"+classes[i])]
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
    train_path = "data_s1/train"
    test_path = "data_s1/test"
    v_path = "data_s1/valid"

    #TODO: change "_batches" to "_gen"
    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=classes, batch_size=bSize)
    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=classes, batch_size=bSize, shuffle=False)
    valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input).flow_from_directory(directory=v_path, target_size=(224,224), classes=classes, batch_size=bSize)

    images, labels = next(train_batches)

    # plotImages function taken from Tensorflow website tutorial
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 10, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    #TODO: Import model:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Activation, Dense, Flatten

    base = VGG16(weights="imagenet")
    model = Sequential()
    for layer in base.layers[:-1]:
        model.add(layer)

    print(model.summary())

    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(units=36, activation="softmax"))

    print(model.summary())

    from tensorflow.keras.optimizers import Adam

    model.compile(optimizer= Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])
    results = model.fit(x=train_batches, validation_data=valid_batches, epochs=EPOCHS, verbose=2)
    model.save("models/model_s1_imagenet_"+CONFIG)

    # model = tf.keras.models.load_model("model_s1_imagenet")
    # results = model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

    # # Visualizing Training and Validation metrics
    # print(results.history.keys())

    # summarize history for accuracy
    plt.plot(results.history['accuracy'], label="Training Accuracy")
    plt.plot(results.history['val_accuracy'], label="Validation Accuracy")
    plt.title('Stage 1 Training/Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/acc/model_s1_imagenet_acc_"+CONFIG+".png")
    plt.close()
    # summarize history for loss
    plt.plot(results.history['loss'], label="Training Loss")
    plt.plot(results.history['val_loss'], label="Validation Loss")
    plt.title('Stage 1 Training/Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/loss/model_s1_imagenet_loss_"+CONFIG+".png")
    plt.close()

    # Stage 1 Results:
    test_images, test_labels = next(test_batches)

    loss, acc = model.evaluate(x=test_batches, batch_size=bSize)

    with open("results/testResults.txt", "a") as file:
        file.write("model_s1_imagenet_" + CONFIG+": \n")
        file.write("Mean Test Loss: " + str(loss)+"\n")
        file.write("Mean Test Accuracy: " + str(acc)+"\n\n")
        file.close()

    print("***END OF STAGE 1***")
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

    #Initialize weights from prior stage:

    transfer = Sequential()
    for layer in model.layers[:-1]:
        transfer.add(layer)

    for layer in transfer.layers:
        layer.trainable = False

    transfer.add(Dense(units=2, activation="softmax"))


    transfer.compile(optimizer=Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])
    results = transfer.fit(x=train_batches, validation_data=valid_batches, epochs=EPOCHS, verbose=2)
    transfer.save("models/model_s2_imagenet_"+CONFIG)

    # Visualizing Training and Validation metrics
    print(results.history.keys())

    # summarize history for accuracy
    plt.plot(results.history['accuracy'], label="Training Accuracy")
    plt.plot(results.history['val_accuracy'], label="Validation Accuracy")
    plt.title('Stage 2 Transfer Training/Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/acc/model_s2_imagenet_acc_"+CONFIG+".png")
    plt.close()
    # summarize history for loss
    plt.plot(results.history['loss'], label="Training Loss")
    plt.plot(results.history['val_loss'], label="Validation Loss")
    plt.title('Stage 2 Transfer Training/Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("results/loss/model_s2_imagenet_loss_"+CONFIG+".png")
    plt.close()


    test_images, test_labels = next(test_batches)
  
    predictions = transfer.predict(x=test_batches, verbose=0)

    loss, acc = transfer.evaluate(x=test_batches, batch_size=bSize)

    with open("results/testResults.txt", "a") as file:
        file.write("model_s2_imagenet_" + CONFIG+": \n")
        file.write("Mean Test Loss: " + str(loss)+"\n")
        file.write("Mean Test Accuracy: " + str(acc)+"\n\n")
        file.close()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import numpy as np

    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    viz = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    viz.plot()
    plt.savefig("results/cm/model_s2_imagenet_cm_"+CONFIG+".png")
    plt.close()


    #TODO: ROC Curve
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_predictions(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plt.savefig("results/roc/model_s2_imagenet_ROC_"+CONFIG+".png")
    plt.close()

    #TODO: Precision-Recall Curve
    from sklearn.metrics import PrecisionRecallDisplay

    PrecisionRecallDisplay.from_predictions(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    plt.savefig("results/pr/model_s2_imagenet_PR_"+CONFIG+".png")
    plt.close()

