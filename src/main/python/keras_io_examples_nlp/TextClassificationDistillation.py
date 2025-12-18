from TextClassificationTorchUtilities import *
from keras import ops

"""
Knowledge Distillation from a saved Keras PyTorch TextClassification model (Teacher) 
to a much smaller Student model.

Imdb data source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
"""

"""
## Setup
"""
imdb_files_dir = "tests/aclImdb2.5K"
epochs = 5


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            ops.softmax(teacher_pred / self.temperature, axis=0),
            ops.softmax(y_pred / self.temperature, axis=0),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)


def buildstudent(train_ds, val_ds, test_ds, max_features=max_features, embedding_dim=128, epochs=3):
    inputs = keras.Input(shape=(None,), dtype="int64")

    x = layers.Embedding(max_features, 32)(inputs)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = keras.Model(inputs, predictions)

    return model


# Load Imdb data
raw_train_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size,
                                           validation_split=0.2, seed=1337, subset="training", format="grain")
raw_val_ds = text_dataset_from_directory(imdb_files_dir+"/train", batch_size=batch_size,
                                         validation_split=0.2, seed=1337, subset="validation", format="grain")
raw_test_ds = text_dataset_from_directory(
    imdb_files_dir+"/test", batch_size=batch_size, format="grain")

# Data mappers from raw data
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

print("Load Vocab")
loadVocabFromFile(SAVE_TO_DIR+"TextClassificationVocab.pkl")

print("Load Teacher")
teacherPath = SAVE_TO_DIR+'TextClassificationTorchModel.keras'
teacher = keras.models.load_model(teacherPath)
teacher.trainable = False

print("Build Student Model")
student = buildstudent(train_ds, val_ds, test_ds,
                       max_features, embedding_dim, epochs)

distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(), metrics=[keras.metrics.BinaryAccuracy()],
    student_loss_fn=keras.losses.BinaryCrossentropy(from_logits=False),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

print("Train Student via Distillation")
# Distill teacher to student

distiller.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate student on test dataset
print("Evaluate Student")
distiller.evaluate(test_ds)

print("Save Model")
student.save(SAVE_TO_DIR+'TextClassificationStudentModel.keras')
