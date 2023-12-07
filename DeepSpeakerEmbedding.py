from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_embedding_network(input_dim, embedding_dim=64):
    # Define the network architecture
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    embeddings = Dense(embedding_dim)(x)

    # Define the model
    model = Model(inputs=inputs, outputs=embeddings)

    return model

def triplet_loss(y_true, y_pred, alpha=0.2):
    # Triplet loss function for training the embedding network
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
    return loss

def train_deep_speaker_embeddings(data, labels, n_epochs=10, batch_size=32):
    # Encode labels to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    # Create the embedding network
    model = create_embedding_network(input_dim=data.shape[1])

    # Compile the model with the triplet loss function
    model.compile(loss=triplet_loss, optimizer=Adam())

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)

    # Test the model
    print("Testing model...")
    y_pred = model.predict(X_test)

    # Calculate the accuracy of our model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return model, le, accuracy
