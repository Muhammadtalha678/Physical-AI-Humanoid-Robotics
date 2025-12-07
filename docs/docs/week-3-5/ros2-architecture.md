---
sidebar_position: 1
title: "ROS 2 Architecture and Core Concepts"
---

# ROS 2 Architecture and Core Concepts

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## What is ROS 2?

ROS 2 is the next generation of the Robot Operating System, designed to address the limitations of ROS 1 and provide enhanced capabilities for modern robotics applications. Unlike ROS 1, ROS 2 is built on DDS (Data Distribution Service) for communication, providing better support for real-time systems, multi-robot systems, and industrial applications.

## Core Architecture

### Client Library Implementations
ROS 2 supports multiple client libraries:
- **rclcpp**: C++ client library
- **rclpy**: Python client library
- **rcl**: C client library (lower level)
- **rclc**: C library for microcontrollers

### DDS/RMW Layer
The ROS Middleware (RMW) layer abstracts the underlying DDS implementation, allowing users to switch between different DDS vendors (Fast DDS, Cyclone DDS, RTI Connext DDS) without changing application code.

### Communication Patterns

#### Publisher-Subscriber (Pub/Sub)
- **Asynchronous**: Publishers and subscribers don't need to be synchronized
- **Topic-based**: Communication happens over named topics
- **Multiple publishers/subscribers**: Multiple nodes can publish to or subscribe from the same topic

#### Client-Service (C/S)
- **Synchronous**: Client waits for response from service
- **Request-Response**: One request generates one response
- **Stateless**: Each request is independent

#### Action
- **Asynchronous with feedback**: Long-running tasks with progress updates
- **Goal/Result/Feeback**: Three-part communication pattern
- **Preemption**: Ability to cancel ongoing actions

## Nodes

Nodes are the fundamental building blocks of a ROS 2 system. Each node:
- Runs a single-threaded or multi-threaded process
- Communicates with other nodes through topics, services, or actions
- Can be written in different programming languages
- Has a unique name within the ROS graph

### Node Lifecycle
ROS 2 provides a lifecycle node concept for complex systems:
- **Unconfigured**: Node created but not configured
- **Inactive**: Configured but not active
- **Active**: Running and operational
- **Finalized**: Node is shutting down

## Topics and Messages

### Topics
- Named buses over which nodes exchange messages
- Anonymous publishing and subscribing
- Support for different Quality of Service (QoS) profiles

### Messages
- Data structures for communication
- Defined in `.msg` files
- Generated in multiple languages
- Strongly typed and serialized

## Services

- Synchronous request-response communication
- Defined in `.srv` files
- Consist of request and response message types
- Useful for tasks that require immediate response

## Actions

- Asynchronous goal-oriented communication
- Defined in `.action` files
- Consist of goal, result, and feedback message types
- Support for preemption and cancellation

## Quality of Service (QoS)

QoS profiles allow fine-tuning of communication behavior:
- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local
- **History**: Keep last N messages vs. keep all
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: How to determine if publisher is alive

## Parameter System

- Dynamic parameter configuration
- Parameter services for getting/setting parameters
- Parameter callbacks for responding to changes
- Support for parameter validation

## Launch System

- XML-based launch files for starting multiple nodes
- Python launch system for complex startup logic
- Parameter passing and node remapping
- Process management and monitoring

## Learning Objectives

After completing this section, you should be able to:
- Explain the core architecture of ROS 2 and its advantages over ROS 1
- Describe the different communication patterns in ROS 2
- Understand the role of nodes, topics, services, and actions
- Identify when to use different communication patterns
- Explain the concept of Quality of Service in ROS 2

## Related Content

For deeper understanding of related concepts, see:
- [Introduction to Physical AI](../week-1-2/introduction-to-physical-ai.md) - For context on AI in physical systems
- [Nodes, topics, services, and actions](./ros2-nodes-topics.md) - Detailed implementation of communication patterns
- [Building ROS 2 packages with Python](./ros2-packages-python.md) - Practical implementation guide
- [Launch files and parameter management](./ros2-launch-files.md) - Advanced configuration techniques
- [NVIDIA Isaac SDK and Isaac Sim](../week-8-10/isaac-sdk-sim.md) - For ROS 2 integration with NVIDIA platforms
- [Gazebo simulation environment setup](../week-6-7/gazebo-setup.md) - For simulation with ROS 2

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/ros2-examples/simple_publisher.py` - Basic publisher node example demonstrating topics
- `docs/static/code-examples/ros2-examples/simple_subscriber.py` - Basic subscriber node example demonstrating topics

## Exercises

1. Create a simple ROS 2 node that publishes "Hello, World!" messages to a topic
2. Research the differences between ROS 1 and ROS 2 architectures
3. Identify three scenarios where each communication pattern (pub/sub, client/service, action) would be most appropriate
4. Run the provided ROS 2 examples to understand basic communication patterns