---
sidebar_position: 1
title: "Integrating GPT Models for Conversational AI"
---

# Integrating GPT Models for Conversational AI

Large Language Models (LLMs) like GPT have revolutionized conversational AI systems. This section covers the integration of GPT models into humanoid robotics systems, enabling natural and contextually aware interactions.

## Introduction to LLMs in Robotics

### Why LLMs for Robotics?

Large Language Models bring several advantages to robotic systems:

- **Natural Language Understanding**: Ability to understand complex, ambiguous, or colloquial language
- **Contextual Reasoning**: Understanding of context, world knowledge, and common sense
- **Flexible Interaction**: Ability to handle unexpected questions and requests
- **Knowledge Integration**: Access to vast amounts of world knowledge

### Challenges in LLM-Robot Integration

- **Real-time Constraints**: LLM inference can be slow
- **Embodied Grounding**: Connecting language to physical actions and environment
- **Safety and Control**: Ensuring robot actions are safe and appropriate
- **Context Management**: Maintaining conversation and task context

## GPT Model Fundamentals

### Architecture Overview

GPT models are based on the Transformer architecture with autoregressive training:

```python
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPTRobotInterface:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_text(self, text):
        """
        Encode text for model input
        """
        return self.tokenizer.encode(text, return_tensors='pt')

    def decode_output(self, token_ids):
        """
        Decode model output to text
        """
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
```

### Prompt Engineering for Robotics

Effective prompting is crucial for robotics applications:

```python
class RobotPromptEngineer:
    def __init__(self):
        self.system_prompt = """
        You are a helpful robotic assistant. You have access to a humanoid robot with the following capabilities:
        - Navigation: can move to locations (go_to(location))
        - Manipulation: can pick up and place objects (pick_up(object), place_at(location))
        - Speech: can speak responses (speak(text))
        - Perception: can recognize objects and people (recognize_object(target))

        Always respond with executable actions for the robot or with a natural language response.
        Be concise, safe, and helpful.
        """

    def create_robot_prompt(self, user_input, robot_state, available_actions):
        """
        Create a contextual prompt for the robot
        """
        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Current robot state: {robot_state}\n"
        prompt += f"Available actions: {available_actions}\n"
        prompt += f"User input: {user_input}\n"
        prompt += "Robot response:"

        return prompt

    def parse_model_response(self, response):
        """
        Parse the model response into robot actions
        """
        # Look for action patterns in the response
        import re

        # Example: Extract navigation commands
        nav_match = re.search(r"go_to\(([^)]+)\)", response)
        if nav_match:
            return {
                'action': 'navigation',
                'target': nav_match.group(1),
                'raw_response': response
            }

        # Example: Extract manipulation commands
        pick_match = re.search(r"pick_up\(([^)]+)\)", response)
        if pick_match:
            return {
                'action': 'manipulation',
                'target': pick_match.group(1),
                'raw_response': response
            }

        # Default: speech response
        return {
            'action': 'speak',
            'text': response,
            'raw_response': response
        }
```

## Context Management

### Conversation History

Maintaining conversation context is essential for coherent interactions:

```python
class ConversationManager:
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.conversation_history = []
        self.context_summary = ""

    def add_turn(self, user_input, robot_response):
        """
        Add a turn to the conversation history
        """
        turn = {
            'user': user_input,
            'robot': robot_response,
            'timestamp': self.get_current_time()
        }
        self.conversation_history.append(turn)

        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

        # Update context summary periodically
        if len(self.conversation_history) % 3 == 0:
            self.update_context_summary()

    def get_context_prompt(self):
        """
        Get the context to include in prompts
        """
        context = "Recent conversation:\n"
        for i, turn in enumerate(self.conversation_history[-5:]):  # Last 5 turns
            context += f"User: {turn['user']}\n"
            context += f"Robot: {turn['robot']}\n"

        if self.context_summary:
            context += f"\nConversation summary: {self.context_summary}\n"

        return context

    def update_context_summary(self):
        """
        Create a summary of the conversation so far
        """
        # In practice, this would use a smaller model to summarize
        # For this example, we'll just concatenate recent topics
        recent_inputs = [turn['user'] for turn in self.conversation_history[-3:]]
        self.context_summary = " ".join(recent_inputs)[:200]  # Truncate to 200 chars
```

### Task and World State Context

Integrating task and world state into the conversation:

```python
class WorldStateContext:
    def __init__(self):
        self.objects = {}
        self.locations = {}
        self.robot_state = {}
        self.current_task = None

    def update_object(self, obj_id, properties):
        """
        Update information about an object
        """
        if obj_id not in self.objects:
            self.objects[obj_id] = {}
        self.objects[obj_id].update(properties)

    def get_world_context(self):
        """
        Get current world state for context
        """
        context = "Current world state:\n"

        # Objects in environment
        if self.objects:
            context += "Visible objects:\n"
            for obj_id, props in self.objects.items():
                context += f"- {obj_id}: {props}\n"

        # Robot state
        if self.robot_state:
            context += f"Robot location: {self.robot_state.get('location', 'unknown')}\n"
            context += f"Robot battery: {self.robot_state.get('battery', 100)}%\n"

        # Current task
        if self.current_task:
            context += f"Current task: {self.current_task}\n"

        return context

    def create_rich_prompt(self, user_input, conversation_context):
        """
        Create a rich prompt with world state and conversation context
        """
        world_context = self.get_world_context()
        full_prompt = f"{world_context}\n{conversation_context}\n"
        full_prompt += f"User request: {user_input}\n"
        full_prompt += "Robot response with action:"

        return full_prompt
```

## Safety and Control Mechanisms

### Action Validation

Ensuring robot actions are safe and appropriate:

```python
class SafetyValidator:
    def __init__(self):
        self.forbidden_actions = [
            'harm', 'dangerous', 'unsafe', 'inappropriate'
        ]
        self.safe_locations = set()
        self.forbidden_objects = set()

    def validate_action(self, action, context):
        """
        Validate that an action is safe to execute
        """
        # Check if action contains forbidden terms
        action_text = str(action).lower()
        for forbidden in self.forbidden_actions:
            if forbidden in action_text:
                return False, f"Action contains forbidden term: {forbidden}"

        # Check location safety
        if 'action' in action and action['action'] == 'navigation':
            target = action.get('target', '')
            if target not in self.safe_locations:
                return False, f"Navigation to {target} is not in safe locations"

        # Check object safety
        if 'action' in action and action['action'] == 'manipulation':
            target = action.get('target', '')
            if target in self.forbidden_objects:
                return False, f"Manipulation of {target} is forbidden"

        # Check for harmful language
        if 'text' in action:
            if self.contains_harmful_language(action['text']):
                return False, "Response contains potentially harmful language"

        return True, "Action is safe"

    def contains_harmful_language(self, text):
        """
        Check if text contains harmful language
        """
        # This would typically use a more sophisticated content filter
        harmful_keywords = [
            'injure', 'hurt', 'damage', 'break', 'destroy',
            'inappropriate', 'offensive', 'harmful'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in harmful_keywords)
```

### Response Filtering

Filtering LLM responses for safety:

```python
class ResponseFilter:
    def __init__(self):
        self.safety_validator = SafetyValidator()

    def filter_response(self, model_response, user_input):
        """
        Filter model response for safety and appropriateness
        """
        # Parse the response into actions
        parsed_action = self.safety_validator.parse_model_response(model_response)

        # Validate the action
        is_safe, reason = self.safety_validator.validate_action(parsed_action, user_input)

        if is_safe:
            return parsed_action
        else:
            # Return a safe fallback response
            return {
                'action': 'speak',
                'text': "I'm sorry, I can't do that. How else can I help you?",
                'raw_response': model_response
            }

    def moderate_content(self, text):
        """
        Moderate content in responses
        """
        # Apply content filtering
        if self.is_inappropriate(text):
            return self.generate_safe_alternative(text)
        return text

    def is_inappropriate(self, text):
        """
        Check if text is inappropriate
        """
        # Simple keyword-based check (in practice, use more sophisticated methods)
        inappropriate_keywords = [
            'inappropriate', 'offensive', 'private', 'personal', 'confidential'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in inappropriate_keywords)

    def generate_safe_alternative(self, original_text):
        """
        Generate a safe alternative to inappropriate content
        """
        return "I can't respond to that. How else can I assist you today?"
```

## Integration Patterns

### Async Processing Pattern

Handling LLM processing without blocking robot operations:

```python
import asyncio
import threading
from queue import Queue

class AsyncGPTProcessor:
    def __init__(self):
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.model_interface = GPTRobotInterface()
        self.is_running = False
        self.worker_thread = None

    def start_processing(self):
        """
        Start the async processing thread
        """
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_requests)
        self.worker_thread.start()

    def _process_requests(self):
        """
        Process requests in the background
        """
        while self.is_running:
            try:
                # Get request from queue
                request = self.request_queue.get(timeout=1.0)

                # Process with GPT model
                response = self.model_interface.generate_response(request)

                # Put response in output queue
                self.response_queue.put(response)

            except:
                continue  # Timeout or other exception, continue loop

    def submit_request(self, prompt):
        """
        Submit a request for async processing
        """
        self.request_queue.put(prompt)

    def get_response(self, timeout=5.0):
        """
        Get response with timeout
        """
        try:
            return self.response_queue.get(timeout=timeout)
        except:
            return None

    def stop_processing(self):
        """
        Stop the processing thread
        """
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join()
```

### Context-Aware Response Generation

Generating responses that consider the robot's capabilities:

```python
class ContextAwareGenerator:
    def __init__(self):
        self.gpt_interface = GPTRobotInterface()
        self.world_context = WorldStateContext()
        self.conversation_manager = ConversationManager()
        self.safety_filter = ResponseFilter()

    def generate_robot_response(self, user_input, robot_capabilities):
        """
        Generate a context-aware response for the robot
        """
        # Create rich context
        conversation_context = self.conversation_manager.get_context_prompt()
        world_context = self.world_context.get_world_context()

        # Create prompt with all context
        full_prompt = self.create_contextual_prompt(
            user_input, conversation_context, world_context, robot_capabilities
        )

        # Generate response with GPT
        raw_response = self.gpt_interface.generate_response(full_prompt)

        # Parse and validate response
        parsed_action = self.safety_filter.filter_response(raw_response, user_input)

        # Update conversation history
        self.conversation_manager.add_turn(user_input, str(parsed_action))

        return parsed_action

    def create_contextual_prompt(self, user_input, conv_context, world_context, capabilities):
        """
        Create a prompt with full contextual information
        """
        prompt = f"System: You are a {capabilities} humanoid robot assistant.\n"
        prompt += f"{world_context}\n"
        prompt += f"{conv_context}\n"
        prompt += f"User: {user_input}\n"
        prompt += "Provide a response with specific robot actions. If you cannot perform the requested action, explain why and suggest alternatives.\n"
        prompt += "Response:"

        return prompt

    def handle_context_switching(self, new_context):
        """
        Handle switching between different interaction contexts
        """
        # Clear or adapt conversation history
        self.conversation_manager.conversation_history = []
        self.conversation_manager.context_summary = ""

        # Update world state
        self.world_context = new_context
```

## Practical Implementation Considerations

### Latency Management

Managing the latency of LLM responses in real-time robotics:

```python
import time

class LatencyManager:
    def __init__(self):
        self.avg_response_time = 2.0  # seconds
        self.max_acceptable_latency = 5.0  # seconds
        self.response_times = []

    def measure_response_time(self, start_time, end_time):
        """
        Measure and track response time
        """
        response_time = end_time - start_time
        self.response_times.append(response_time)

        # Update average (simple moving average)
        if len(self.response_times) > 10:  # Keep last 10 measurements
            self.response_times = self.response_times[-10:]

        self.avg_response_time = sum(self.response_times) / len(self.response_times)

    def should_use_llm(self):
        """
        Decide whether to use LLM based on current system load
        """
        return self.avg_response_time < self.max_acceptable_latency

    def generate_fallback_response(self, user_input):
        """
        Generate a quick fallback response if LLM is too slow
        """
        simple_responses = {
            'hello': 'Hello! How can I help you?',
            'help': 'I can help with navigation, object manipulation, and information.',
            'how are you': 'I am functioning well, thank you for asking!',
            'what can you do': 'I can navigate, manipulate objects, and have conversations.'
        }

        user_lower = user_input.lower()
        for key, response in simple_responses.items():
            if key in user_lower:
                return response

        return "I'm processing your request. Please wait a moment."
```

### Memory Management

Managing memory for continuous conversation:

```python
class MemoryManager:
    def __init__(self, max_memory_mb=1000):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory_usage = 0
        self.conversation_segments = []

    def add_conversation_segment(self, segment):
        """
        Add a conversation segment to memory
        """
        import sys
        segment_size = sys.getsizeof(str(segment))

        if self.current_memory_usage + segment_size > self.max_memory:
            # Remove oldest segments to make space
            self.prune_memory(segment_size)

        self.conversation_segments.append(segment)
        self.current_memory_usage += segment_size

    def prune_memory(self, needed_space):
        """
        Prune memory to make room for new data
        """
        while (self.current_memory_usage > self.max_memory * 0.7 or
               needed_space > self.max_memory - self.current_memory_usage) and \
              len(self.conversation_segments) > 1:
            removed_segment = self.conversation_segments.pop(0)
            import sys
            removed_size = sys.getsizeof(str(removed_segment))
            self.current_memory_usage -= removed_size

    def compress_context(self):
        """
        Compress conversation context to save memory
        """
        # Summarize long conversation histories
        if len(self.conversation_segments) > 20:
            # Keep recent segments, summarize older ones
            recent = self.conversation_segments[-5:]
            older = self.conversation_segments[:-5]

            # Create summary of older segments
            summary = self.summarize_segments(older)

            self.conversation_segments = [summary] + recent
            self.current_memory_usage = sum(
                sys.getsizeof(str(seg)) for seg in self.conversation_segments
            )

    def summarize_segments(self, segments):
        """
        Summarize a list of conversation segments
        """
        # Simple summary - in practice, use a summarization model
        all_text = " ".join(str(seg) for seg in segments)
        return f"Earlier conversation summary: {all_text[:200]}..."
```

## Learning Objectives

After completing this section, you should be able to:
- Integrate GPT models with humanoid robotics systems
- Implement context management for conversational AI
- Apply safety and validation mechanisms for LLM outputs
- Design async processing patterns for real-time robotics
- Create context-aware response generation systems
- Manage latency and memory constraints in LLM integration

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/capstone/humanoid_capstone_template.py` - Template for humanoid robot integration

## Exercises

1. Implement a GPT interface for a humanoid robot
2. Create a context management system for continuous conversations
3. Design safety validation mechanisms for LLM-generated robot actions
4. Implement async processing for LLM responses
5. Create a memory management system for long-term interactions