import numpy as np
import re


class OSM_0_1:
    def __init__(
        self,
        input_embedding_lookup: dict,
        query_embedding_matrix: list,
        key_embedding_matrix: list,
        value_embedding_matrix: list,
        output_embedding_matrix: list,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        model_id: str,
        vocab: list,
    ):
        self.input_embedding_lookup = input_embedding_lookup
        self.query_embedding_matrix = np.array(query_embedding_matrix)
        self.key_embedding_matrix = np.array(key_embedding_matrix)
        self.value_embedding_matrix = np.array(value_embedding_matrix)
        self.output_embedding_matrix = np.array(output_embedding_matrix)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.model_id = model_id
        self.vocab = vocab

    def generate_word(
        self, input_text: str, temperature: float, attention_repition: int
    ) -> list:
        """
        Generates a word based on the input text.

        Args:
            input_text (str): The input text to generate the word from.
            temperature (float): The temperature of the softmax function, a value between 0 and 1.
            attention_repition (int): The number of times to repeat the attention mechanism.

        Returns:
            list: The probabilities of the generated word.
        """

        if input_text is None:
            raise ValueError("Input text is None")
        if temperature <= 0:
            raise ValueError("Temperature is 0 or negative")
        if attention_repition <= 0:
            raise ValueError("Attention repetition is negative")

        print("Input text:", input_text)
        print("Temperature:", temperature)
        print("Attention Repetition:", attention_repition)

        try:
            embedded_input = self.input_embedding(
                input_text, self.input_embedding_lookup
            )
        except KeyError as e:
            print(f"KeyError: {e}")
            raise

        new_embedded_input = []

        for i in embedded_input:
            new_i = [len(i) - 1]
            for j in i:
                new_i.append(j[0])
            new_embedded_input.append(new_i)

        embedded_input = new_embedded_input

        print("Length of embedded_input:", len(embedded_input))

        for _ in range(attention_repition):
            try:
                current_output = self.attention(
                    embedded_input,
                    self.query_embedding_matrix,
                    self.key_embedding_matrix,
                    self.value_embedding_matrix,
                    temperature,
                )
            except Exception as e:
                print(f"Exception in attention: {e}")
                raise

        print("Length of current_output:", len(current_output))

        output = self.output_embedding(current_output, self.output_embedding_matrix)
        output_weights = self.softmax(output[len(output) - 1], temperature)

        print("Length of output:", len(output))
        print("Length of output_weights:", len(output_weights))

        return output_weights

    def input_embedding(self, input_text: str, input_embeddings: dict):
        tokens = re.findall(r"\b\w+\b|[^\w\s{}|]|[{|ENDOFTEXT|}]", input_text.lower())

        if input_embeddings is None:
            raise ValueError("Input embeddings is None")

        embeddings = [
            input_embeddings.get(token, [0] * self.embedding_dim) for token in tokens
        ]
        return embeddings

    def softmax(self, arr: list, temperature: float):
        if temperature == 0:
            raise ValueError("Temperature cannot be 0")

        try:
            return np.exp(np.array(arr) / temperature) / np.sum(
                np.exp(np.array(arr) / temperature)
            )
        except Exception as e:
            print(f"Exception in softmax: {e}")
            raise

    def get_queries(self, input_seq: list, query_embeddings: list):
        if query_embeddings is None:
            raise ValueError("Query embeddings is None")

        return [np.matmul(query_embeddings, input_vec) for input_vec in input_seq]

    def get_keys(self, input_seq: list, key_embeddings: list):
        if key_embeddings is None:
            raise ValueError("Key embeddings is None")

        return [np.matmul(key_embeddings, input_vec) for input_vec in input_seq]

    def calculate_attention(self, queries: list, keys: list):
        return [[query * key for key in keys] for query in queries]

    def normalize_attention(self, attention: list, temperature: float):
        if attention is None:
            raise ValueError("Attention is None")

        normalized_attention = []
        for row in attention:
            normalized_row = []
            for element in row:
                normalized_row.append(element / temperature)
            normalized_attention.append(normalized_row)
        return normalized_attention

    def apply_attention(self, input_seq: list, attention: list, value_embeddings: list):
        if value_embeddings is None:
            raise ValueError("Value embeddings is None")
        if attention is None:
            raise ValueError("Attention is None")

        updated_attention = []
        for attention_row, input_vec in zip(attention, input_seq):
            updated_attention_row = []
            for attention_element, value_embedding in zip(
                attention_row, value_embeddings
            ):
                updated_attention_row.append(
                    attention_element * np.matmul(value_embedding, input_vec)
                )
            updated_attention.append(updated_attention_row)

        return updated_attention

    def attention(
        self,
        input_seq: list,
        query_embeddings: list,
        key_embeddings: list,
        value_embeddings: list,
        temperature: list,
    ):
        if query_embeddings is None:
            raise ValueError("Query embeddings is None")
        if key_embeddings is None:
            raise ValueError("Key embeddings is None")
        if value_embeddings is None:
            raise ValueError("Value embeddings is None")

        queries = self.get_queries(input_seq, query_embeddings)
        keys = self.get_keys(input_seq, key_embeddings)
        attention = self.calculate_attention(queries, keys)
        normalized_attention = self.normalize_attention(attention, temperature)
        output = self.apply_attention(input_seq, normalized_attention, value_embeddings)

        return output

    def output_embedding(self, current_output: list, output_embeddings: list):
        if output_embeddings is None:
            raise ValueError("Output embeddings is None")

        return np.matmul(output_embeddings, current_output)

    def backward_pass(
        self, embedded_input, output_weights, target_one_hot, temperature
    ):
        """
        Performs the backward pass and computes gradients.

        Args:
            embedded_input (list): The embedded input sequence.
            output_weights (list): The output probabilities from the forward pass.
            target_one_hot (list): One-hot encoded target word.
            temperature (float): Temperature parameter for the softmax function.

        Returns:
            dict: Gradients for each embedding matrix.
        """
        gradients = {
            "query_embedding_matrix": np.zeros_like(self.query_embedding_matrix),
            "key_embedding_matrix": np.zeros_like(self.key_embedding_matrix),
            "value_embedding_matrix": np.zeros_like(self.value_embedding_matrix),
            "output_embedding_matrix": np.zeros_like(self.output_embedding_matrix),
        }

        d_output = output_weights - target_one_hot
        d_output_embedding_matrix = np.outer(d_output, embedded_input[-1])

        gradients["output_embedding_matrix"] = d_output_embedding_matrix

        d_attention = np.matmul(d_output, self.output_embedding_matrix.T)
        d_normalized_attention = d_attention / temperature

        for i in range(len(embedded_input)):
            d_queries = d_normalized_attention * self.key_embedding_matrix
            d_keys = d_normalized_attention * self.query_embedding_matrix
            d_values = d_normalized_attention * embedded_input[i]

            gradients["query_embedding_matrix"] += np.outer(
                d_queries, embedded_input[i]
            )
            gradients["key_embedding_matrix"] += np.outer(d_keys, embedded_input[i])
            gradients["value_embedding_matrix"] += np.outer(d_values, embedded_input[i])

        return gradients

    def update_parameters(self, gradients, learning_rate):
        """
        Updates the parameters using the computed gradients.

        Args:
            gradients (dict): Gradients for each embedding matrix.
            learning_rate (float): Learning rate for the parameter updates.
        """
        self.query_embedding_matrix -= (
            learning_rate * gradients["query_embedding_matrix"]
        )
        self.key_embedding_matrix -= learning_rate * gradients["key_embedding_matrix"]
        self.value_embedding_matrix -= (
            learning_rate * gradients["value_embedding_matrix"]
        )
        self.output_embedding_matrix -= (
            learning_rate * gradients["output_embedding_matrix"]
        )

    def train_step(self, input_text: str, target_word: str, learning_rate: float):
        """
        Performs a single training step.

        Args:
            input_text (str): The input text.
            target_word (str): The target word.
            learning_rate (float): The learning rate for updating the weights.

        Returns:
            float: The loss value for this training step.
        """
        # Forward pass
        embedded_input = self.input_embedding(input_text, self.input_embedding_lookup)
        output_weights = self.generate_word(
            input_text, temperature=1.0, attention_repition=1
        )

        target_idx = self.vocab.index(target_word)
        target_one_hot = np.zeros(self.vocab_size)
        target_one_hot[target_idx] = 1

        # Calculate loss (cross-entropy)
        loss = -np.sum(target_one_hot * np.log(output_weights))

        # Backward pass
        gradients = self.backward_pass(
            embedded_input, output_weights, target_one_hot, temperature=1.0
        )

        # Update weights using gradients
        self.update_parameters(gradients, learning_rate)

        return loss

    def train(self, training_data: list, epochs: int, learning_rate: float):
        """
        Trains the model over multiple epochs.

        Args:
            training_data (list): A list of tuples, where each tuple contains an input text and the target word.
            epochs (int): The number of epochs to train the model.
            learning_rate (float): The learning rate for updating the weights.
        """
        for epoch in range(epochs):
            total_loss = 0
            for input_text, target_word in training_data:
                loss = self.train_step(input_text, target_word, learning_rate)
                total_loss += loss
            average_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}")
