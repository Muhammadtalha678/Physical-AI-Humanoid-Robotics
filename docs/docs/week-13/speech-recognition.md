---
sidebar_position: 2
title: "Speech Recognition and Natural Language Understanding"
---

# Speech Recognition and Natural Language Understanding

Robust speech recognition and natural language understanding (NLU) are fundamental for natural human-robot interaction. This section covers the principles, techniques, and implementation strategies for processing human speech in real-world robotic environments.

## Fundamentals of Speech Processing

### Speech Recognition Pipeline

The speech recognition process involves multiple stages:

```python
import numpy as np
import librosa
import webrtcvad
from scipy import signal

class SpeechRecognitionPipeline:
    def __init__(self):
        self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 2
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

    def preprocess_audio(self, audio_data):
        """
        Preprocess audio for speech recognition
        """
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))

        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

        # Noise reduction
        audio_data = self.apply_noise_reduction(audio_data)

        return audio_data

    def voice_activity_detection(self, audio_data):
        """
        Detect voice activity in audio
        """
        # Convert to frames
        frames = self.frame_audio(audio_data)

        # Apply VAD to each frame
        vad_results = []
        for frame in frames:
            is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
            vad_results.append(is_speech)

        return vad_results

    def frame_audio(self, audio_data):
        """
        Split audio into frames for VAD
        """
        # Pad audio to ensure proper framing
        padding = self.frame_size - (len(audio_data) % self.frame_size)
        padded_audio = np.pad(audio_data, (0, padding), mode='constant')

        # Reshape into frames
        num_frames = len(padded_audio) // self.frame_size
        frames = padded_audio[:num_frames * self.frame_size].reshape(num_frames, self.frame_size)

        return frames.astype(np.int16)

    def apply_noise_reduction(self, audio_data):
        """
        Apply basic noise reduction
        """
        # Spectral subtraction method
        # Calculate noise profile from beginning of audio (assumed to be silence)
        noise_profile = np.mean(np.abs(audio_data[:int(self.sample_rate * 0.5)]))

        # Apply spectral subtraction
        magnitude = np.abs(audio_data)
        enhanced_magnitude = np.maximum(magnitude - noise_profile, 0)
        enhanced_audio = audio_data * (enhanced_magnitude / magnitude)

        return enhanced_audio
```

### Acoustic Models

Understanding different acoustic model approaches:

```python
class AcousticModelManager:
    def __init__(self):
        self.models = {
            'conformer': self.load_conformer_model,
            'wavenet': self.load_wavenet_model,
            'rnn_transducer': self.load_rnnt_model
        }
        self.current_model = None

    def load_conformer_model(self):
        """
        Load Conformer-based acoustic model
        """
        # In practice, this would load a pre-trained model
        # Conformer models provide good accuracy with reasonable latency
        pass

    def load_wavenet_model(self):
        """
        Load WaveNet-based acoustic model
        """
        # WaveNet provides high-quality audio processing but is computationally expensive
        pass

    def load_rnnt_model(self):
        """
        Load RNN-Transducer model
        """
        # RNN-T models are good for streaming recognition
        pass

    def select_model(self, requirements):
        """
        Select appropriate model based on requirements
        """
        if requirements.get('low_latency', False):
            return self.models['rnnt_transducer']()
        elif requirements.get('high_accuracy', False):
            return self.models['conformer']()
        elif requirements.get('streaming', False):
            return self.models['rnnt_transducer']()
        else:
            return self.models['conformer']()
```

## Advanced Speech Recognition

### Online/Streaming Recognition

Real-time speech recognition for continuous interaction:

```python
import asyncio
import threading
from collections import deque

class StreamingSpeechRecognizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.audio_buffer = deque(maxlen=16000)  # 1 second buffer at 16kHz
        self.is_listening = False
        self.recognition_thread = None
        self.partial_results = []
        self.final_results = []

    def start_streaming(self):
        """
        Start streaming speech recognition
        """
        self.is_listening = True
        self.recognition_thread = threading.Thread(target=self._process_stream)
        self.recognition_thread.start()

    def _process_stream(self):
        """
        Process audio stream in real-time
        """
        while self.is_listening:
            if len(self.audio_buffer) >= 1600:  # Process 100ms chunks
                chunk = list(self.audio_buffer)[:1600]
                self.audio_buffer = deque(list(self.audio_buffer)[1600:], maxlen=16000)

                # Perform recognition on chunk
                partial_result = self.recognize_chunk(chunk)

                # Add to partial results
                self.partial_results.append(partial_result)

    def recognize_chunk(self, audio_chunk):
        """
        Recognize a small chunk of audio
        """
        # Apply acoustic model to chunk
        # This is a simplified representation
        # In practice, this would use a streaming model
        return self.apply_acoustic_model(audio_chunk)

    def apply_acoustic_model(self, audio_chunk):
        """
        Apply acoustic model to audio chunk
        """
        # Convert to features
        features = self.extract_features(audio_chunk)

        # Apply model
        # model_output = self.acoustic_model.predict(features)

        # Decode to text
        # text = self.decoder.decode(model_output)

        # For this example, return a placeholder
        return "partial recognition result"

    def get_partial_result(self):
        """
        Get current partial recognition result
        """
        return " ".join(self.partial_results)

    def stop_streaming(self):
        """
        Stop streaming recognition
        """
        self.is_listening = False
        if self.recognition_thread:
            self.recognition_thread.join()
```

### Multi-Microphone Processing

Using multiple microphones for improved speech recognition:

```python
import numpy as np
from scipy import signal

class MultiMicrophoneProcessor:
    def __init__(self, num_mics=4):
        self.num_mics = num_mics
        self.mic_positions = self.calculate_mic_positions()
        self.beamformer = self.initialize_beamformer()

    def calculate_mic_positions(self):
        """
        Calculate microphone positions in array
        """
        # Assume a circular microphone array
        positions = []
        radius = 0.05  # 5cm radius
        for i in range(self.num_mics):
            angle = 2 * np.pi * i / self.num_mics
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append([x, y, 0])
        return np.array(positions)

    def delay_and_sum_beamforming(self, multi_channel_audio, look_direction):
        """
        Apply delay-and-sum beamforming
        """
        # Calculate time delays for target direction
        delays = self.calculate_delays(look_direction)

        # Apply delays to each channel
        delayed_signals = []
        for i, audio in enumerate(multi_channel_audio):
            delayed = self.apply_delay(audio, delays[i])
            delayed_signals.append(delayed)

        # Sum all channels
        beamformed = np.sum(delayed_signals, axis=0)

        return beamformed

    def calculate_delays(self, look_direction):
        """
        Calculate delays for beamforming in target direction
        """
        delays = []
        for pos in self.mic_positions:
            # Calculate distance to plane wave
            distance = np.dot(pos, look_direction)
            delay_samples = distance / 343.0 * 16000  # speed of sound = 343 m/s
            delays.append(int(delay_samples))
        return delays

    def apply_delay(self, signal, delay_samples):
        """
        Apply delay to signal
        """
        if delay_samples > 0:
            return np.concatenate([np.zeros(delay_samples), signal[:-delay_samples]])
        else:
            delay_samples = abs(delay_samples)
            return np.concatenate([signal[delay_samples:], np.zeros(delay_samples)])

    def noise_reduction(self, multi_channel_audio):
        """
        Apply noise reduction using multiple microphones
        """
        # Calculate spatial covariance matrix
        cov_matrix = self.calculate_covariance_matrix(multi_channel_audio)

        # Apply MUSIC or other spatial filtering algorithm
        enhanced_audio = self.apply_spatial_filter(cov_matrix, multi_channel_audio)

        return enhanced_audio

    def calculate_covariance_matrix(self, multi_channel_audio):
        """
        Calculate spatial covariance matrix
        """
        # Convert to frequency domain
        freq_domain = np.array([np.fft.fft(channel) for channel in multi_channel_audio])

        # Calculate covariance matrix for each frequency bin
        cov_matrix = np.zeros((self.num_mics, self.num_mics), dtype=complex)
        for f in range(freq_domain.shape[1]):
            freq_vec = freq_domain[:, f]
            cov_matrix += np.outer(freq_vec, np.conj(freq_vec))

        return cov_matrix / freq_domain.shape[1]
```

## Natural Language Understanding (NLU)

### Intent Classification

Classifying user intents from recognized speech:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re

class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        self.intents = {}
        self.is_trained = False

    def train(self, training_data):
        """
        Train intent classifier
        """
        texts = [item['text'] for item in training_data]
        labels = [item['intent'] for item in training_data]

        self.pipeline.fit(texts, labels)
        self.is_trained = True

        # Store intent definitions
        for item in training_data:
            if item['intent'] not in self.intents:
                self.intents[item['intent']] = []
            self.intents[item['intent']].append(item['text'])

    def classify_intent(self, text):
        """
        Classify intent of input text
        """
        if not self.is_trained:
            return {'intent': 'unknown', 'confidence': 0.0}

        # Predict intent
        predicted_intent = self.pipeline.predict([text])[0]
        confidence = max(self.pipeline.predict_proba([text])[0])

        return {
            'intent': predicted_intent,
            'confidence': confidence,
            'entities': self.extract_entities(text, predicted_intent)
        }

    def extract_entities(self, text, intent):
        """
        Extract entities based on intent
        """
        entities = []

        # Define entity patterns for different intents
        entity_patterns = {
            'navigation': [
                (r'to (\w+)', 'location'),
                (r'go to (\w+)', 'location'),
                (r'navigate to (\w+)', 'location')
            ],
            'manipulation': [
                (r'pick up (\w+)', 'object'),
                (r'grasp (\w+)', 'object'),
                (r'take (\w+)', 'object')
            ],
            'information': [
                (r'what is (\w+)', 'topic'),
                (r'tell me about (\w+)', 'topic'),
                (r'explain (\w+)', 'topic')
            ]
        }

        if intent in entity_patterns:
            for pattern, entity_type in entity_patterns[intent]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'type': entity_type,
                        'value': match,
                        'text': match
                    })

        return entities
```

### Context-Aware Understanding

Understanding speech in context:

```python
class ContextAwareNLU:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.context_stack = []
        self.entity_linker = EntityLinker()  # Assumed implementation

    def process_utterance(self, text, context=None):
        """
        Process utterance with context awareness
        """
        # Add context to text for better understanding
        contextual_text = self.add_context_to_text(text, context)

        # Classify intent
        intent_result = self.intent_classifier.classify_intent(contextual_text)

        # Resolve entities in context
        resolved_entities = self.resolve_entities_in_context(
            intent_result['entities'], context
        )

        # Update context
        self.update_context(intent_result, resolved_entities)

        return {
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'entities': resolved_entities,
            'context': self.get_current_context()
        }

    def add_context_to_text(self, text, context):
        """
        Add relevant context to text for better understanding
        """
        if not context:
            return text

        # Add context-relevant information to text
        context_text = ""
        if 'previous_intent' in context:
            context_text += f" previously intent was {context['previous_intent']}. "
        if 'current_task' in context:
            context_text += f" current task is {context['current_task']}. "
        if 'available_objects' in context:
            context_text += f" available objects are {', '.join(context['available_objects'])}. "

        return context_text + text

    def resolve_entities_in_context(self, entities, context):
        """
        Resolve entities based on context
        """
        resolved = []
        for entity in entities:
            if entity['type'] == 'location' and context:
                # Resolve ambiguous locations
                resolved_value = self.resolve_location(entity['value'], context)
                resolved.append({
                    'type': entity['type'],
                    'value': resolved_value,
                    'original_text': entity['text']
                })
            elif entity['type'] == 'object' and context:
                # Resolve object references
                resolved_value = self.resolve_object(entity['value'], context)
                resolved.append({
                    'type': entity['type'],
                    'value': resolved_value,
                    'original_text': entity['text']
                })
            else:
                resolved.append(entity)

        return resolved

    def resolve_location(self, location_name, context):
        """
        Resolve location reference in context
        """
        if context and 'known_locations' in context:
            # Check for partial matches or aliases
            for known_loc in context['known_locations']:
                if location_name.lower() in known_loc.lower() or \
                   known_loc.lower() in location_name.lower():
                    return known_loc
        return location_name

    def resolve_object(self, object_name, context):
        """
        Resolve object reference in context
        """
        if context and 'visible_objects' in context:
            # Check for visible objects
            for obj in context['visible_objects']:
                if object_name.lower() in obj.lower():
                    return obj
        return object_name

    def update_context(self, intent_result, entities):
        """
        Update conversation context based on current utterance
        """
        self.context_stack.append({
            'timestamp': self.get_current_time(),
            'intent': intent_result['intent'],
            'entities': entities
        })

        # Keep only recent context
        if len(self.context_stack) > 10:
            self.context_stack.pop(0)

    def get_current_context(self):
        """
        Get current context for the conversation
        """
        if not self.context_stack:
            return {}

        # Return recent context
        recent_context = self.context_stack[-3:]  # Last 3 exchanges
        return {
            'recent_intents': [item['intent'] for item in recent_context],
            'recent_entities': [item['entities'] for item in recent_context],
            'full_context': self.context_stack
        }
```

## Robustness Techniques

### Error Handling and Recovery

Handling recognition errors gracefully:

```python
class RobustSpeechProcessor:
    def __init__(self):
        self.confidence_threshold = 0.7
        self.retry_count = 3
        self.confirmation_required_intents = [
            'navigation', 'manipulation', 'sensitive_action'
        ]

    def process_with_error_handling(self, audio_input):
        """
        Process speech with error handling
        """
        # Initial recognition
        result = self.recognize_speech(audio_input)

        # Check confidence
        if result['confidence'] < self.confidence_threshold:
            # Try alternative recognition approaches
            result = self.retry_with_alternatives(audio_input)

        # If still low confidence, ask for clarification
        if result['confidence'] < self.confidence_threshold:
            return self.request_clarification(result)

        # For critical intents, request confirmation
        if result['intent'] in self.confirmation_required_intents:
            return self.request_confirmation(result)

        return result

    def retry_with_alternatives(self, audio_input):
        """
        Retry recognition with different models or parameters
        """
        # Try with different acoustic models
        models_to_try = ['model1', 'model2', 'model3']
        results = []

        for model in models_to_try:
            result = self.recognize_with_model(audio_input, model)
            results.append(result)

        # Return the result with highest confidence
        best_result = max(results, key=lambda x: x.get('confidence', 0))
        return best_result

    def request_clarification(self, low_confidence_result):
        """
        Request user to clarify their request
        """
        return {
            'action': 'request_clarification',
            'original_result': low_confidence_result,
            'message': "I didn't quite understand. Could you please repeat that?"
        }

    def request_confirmation(self, high_confidence_result):
        """
        Request confirmation for critical actions
        """
        intent = high_confidence_result['intent']
        entities = high_confidence_result['entities']

        if intent == 'navigation':
            location = entities[0]['value'] if entities else 'unknown'
            confirmation_text = f"You want me to go to {location}. Is that correct?"
        elif intent == 'manipulation':
            obj = entities[0]['value'] if entities else 'unknown object'
            confirmation_text = f"You want me to pick up {obj}. Is that correct?"
        else:
            confirmation_text = f"You want me to {intent}. Is that correct?"

        return {
            'action': 'request_confirmation',
            'original_result': high_confidence_result,
            'confirmation_text': confirmation_text
        }

    def handle_confirmation_response(self, user_response):
        """
        Handle user's confirmation response
        """
        if any(word in user_response.lower() for word in ['yes', 'correct', 'right', 'okay', 'sure']):
            # Extract and return the original action
            return self.extract_original_action()
        elif any(word in user_response.lower() for word in ['no', 'wrong', 'incorrect', 'cancel']):
            return {
                'action': 'cancel',
                'message': "Okay, I won't do that."
            }
        else:
            # Unclear response, ask again
            return {
                'action': 'request_clarification',
                'message': "Please say yes to confirm or no to cancel."
            }
```

### Domain Adaptation

Adapting speech recognition to specific domains:

```python
class DomainAdaptiveRecognizer:
    def __init__(self):
        self.base_model = None
        self.domain_models = {}
        self.domain_vocabulary = {}
        self.current_domain = 'general'

    def set_domain(self, domain_name):
        """
        Set the current domain for recognition
        """
        self.current_domain = domain_name

        # Load domain-specific model if available
        if domain_name in self.domain_models:
            self.active_model = self.domain_models[domain_name]
        else:
            self.active_model = self.base_model

        # Update vocabulary for domain
        if domain_name in self.domain_vocabulary:
            self.update_recognition_vocabulary(self.domain_vocabulary[domain_name])

    def add_domain_data(self, domain_name, training_data, vocabulary):
        """
        Add training data and vocabulary for a domain
        """
        # Fine-tune model for domain
        domain_model = self.fine_tune_model(self.base_model, training_data)
        self.domain_models[domain_name] = domain_model

        # Store domain vocabulary
        self.domain_vocabulary[domain_name] = vocabulary

    def fine_tune_model(self, base_model, training_data):
        """
        Fine-tune base model with domain data
        """
        # This would typically involve transfer learning techniques
        # For this example, we'll return the base model
        return base_model

    def update_recognition_vocabulary(self, vocabulary):
        """
        Update the recognition vocabulary for better accuracy
        """
        # Update language model with domain-specific vocabulary
        # This would typically involve updating the decoding graph
        pass

    def recognize_in_domain(self, audio_input):
        """
        Recognize speech in the current domain
        """
        # Use domain-specific model and vocabulary
        result = self.active_model.recognize(audio_input)

        # Apply domain-specific post-processing
        result = self.apply_domain_post_processing(result, self.current_domain)

        return result

    def apply_domain_post_processing(self, result, domain):
        """
        Apply domain-specific post-processing to recognition result
        """
        if domain == 'navigation':
            # Correct common navigation-related misrecognitions
            result['text'] = self.correct_navigation_terms(result['text'])
        elif domain == 'manipulation':
            # Correct manipulation-related terms
            result['text'] = self.correct_manipulation_terms(result['text'])

        return result

    def correct_navigation_terms(self, text):
        """
        Correct common navigation-related misrecognitions
        """
        corrections = {
            'kitchen': ['chicken', 'ketchen', 'citchen'],
            'bedroom': ['bedroom', 'bed room', 'bed rum'],
            'living room': ['living room', 'livingrum', 'living room'],
            'bathroom': ['bathroom', 'bath room', 'bathrum']
        }

        for correct, possible in corrections.items():
            for wrong in possible:
                if wrong.lower() in text.lower():
                    text = text.lower().replace(wrong.lower(), correct)

        return text

    def correct_manipulation_terms(self, text):
        """
        Correct common manipulation-related misrecognitions
        """
        corrections = {
            'cup': ['cup', 'cop', 'cap'],
            'box': ['box', 'bogs', 'books'],
            'bottle': ['bottle', 'battle', 'bodle']
        }

        for correct, possible in corrections.items():
            for wrong in possible:
                if wrong.lower() in text.lower():
                    text = text.lower().replace(wrong.lower(), correct)

        return text
```

## Integration with Robotics Systems

### Robot Speech Interface

Integrating speech recognition with robot control:

```python
class RobotSpeechInterface:
    def __init__(self):
        self.speech_recognizer = StreamingSpeechRecognizer(model_path='model.pt')
        self.nlu_system = ContextAwareNLU()
        self.robot_controller = RobotController()  # Assumed implementation
        self.response_generator = ResponseGenerator()  # Assumed implementation

    def process_speech_command(self, audio_input, robot_state):
        """
        Process speech command and execute robot action
        """
        # Recognize speech
        recognition_result = self.speech_recognizer.recognize(audio_input)

        # Understand intent and entities
        context = {
            'robot_state': robot_state,
            'current_task': self.get_current_task(),
            'available_objects': self.get_visible_objects(),
            'known_locations': self.get_known_locations()
        }

        nlu_result = self.nlu_system.process_utterance(
            recognition_result['text'], context
        )

        # Validate action safety
        if self.is_safe_action(nlu_result):
            # Execute action
            execution_result = self.execute_robot_action(nlu_result)

            # Generate response
            response = self.response_generator.generate_response(
                nlu_result, execution_result
            )

            return {
                'success': True,
                'action': nlu_result,
                'response': response,
                'execution_result': execution_result
            }
        else:
            # Generate safety warning response
            safety_response = self.response_generator.generate_safety_response(nlu_result)
            return {
                'success': False,
                'action': nlu_result,
                'response': safety_response,
                'error': 'Action not safe to execute'
            }

    def is_safe_action(self, action):
        """
        Check if robot action is safe to execute
        """
        # Check various safety constraints
        intent = action['intent']
        entities = action['entities']

        if intent == 'navigation':
            target_location = entities[0]['value'] if entities else None
            return self.is_safe_navigation(target_location)

        elif intent == 'manipulation':
            target_object = entities[0]['value'] if entities else None
            return self.is_safe_manipulation(target_object)

        return True  # Default to safe for other actions

    def execute_robot_action(self, action):
        """
        Execute robot action based on NLU result
        """
        intent = action['intent']
        entities = action['entities']

        if intent == 'navigation':
            target = entities[0]['value'] if entities else None
            return self.robot_controller.navigate_to(target)

        elif intent == 'manipulation':
            target = entities[0]['value'] if entities else None
            return self.robot_controller.manipulate_object(target)

        elif intent == 'speak':
            text = entities[0]['value'] if entities else action['original_text']
            return self.robot_controller.speak(text)

        else:
            # Default action - speak response
            response = self.response_generator.generate_default_response(action)
            return self.robot_controller.speak(response)

    def start_continuous_listening(self):
        """
        Start continuous speech recognition
        """
        self.speech_recognizer.start_listening()

        while True:
            # Get speech input
            audio = self.speech_recognizer.get_audio_input()

            # Process if speech detected
            if self.speech_recognizer.is_speech_detected(audio):
                result = self.process_speech_command(audio, self.robot_controller.get_state())

                # Execute response
                self.robot_controller.execute_response(result['response'])

    def get_current_task(self):
        """
        Get the robot's current task
        """
        return self.robot_controller.get_current_task()

    def get_visible_objects(self):
        """
        Get objects currently visible to the robot
        """
        return self.robot_controller.get_visible_objects()

    def get_known_locations(self):
        """
        Get locations known to the robot
        """
        return self.robot_controller.get_known_locations()
```

## Learning Objectives

After completing this section, you should be able to:
- Implement speech recognition pipelines for robotic applications
- Apply noise reduction and beamforming techniques
- Perform natural language understanding with intent classification
- Handle recognition errors and implement robustness techniques
- Adapt speech recognition to specific robotic domains
- Integrate speech recognition with robot control systems

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/capstone/humanoid_capstone_template.py` - Template for humanoid robot integration

## Exercises

1. Implement a streaming speech recognition system
2. Create an intent classification system for robot commands
3. Design noise reduction algorithms for multi-microphone arrays
4. Implement context-aware natural language understanding
5. Develop error handling and recovery mechanisms for speech recognition