---
sidebar_position: 2
title: "ROS 2 Nodes, Topics, Services, and Actions"
---

# ROS 2 Nodes, Topics, Services, and Actions

This section covers the fundamental communication mechanisms in ROS 2: nodes, topics, services, and actions. Understanding these concepts is crucial for developing distributed robot applications.

## Nodes

### Definition
A node is an executable that uses ROS to communicate with other nodes. Nodes are the basic computational elements of a ROS system where processes take place.

### Node Characteristics
- **Unique Names**: Each node must have a unique name within the ROS graph
- **Namespace Support**: Nodes can be organized in namespaces for better organization
- **Multiple Languages**: Nodes can be written in different programming languages
- **Process Isolation**: Each node runs in its own process

### Node Creation and Management
- **Initialization**: Nodes must be initialized with a name and context
- **Execution**: Nodes typically run in a spin loop to process callbacks
- **Cleanup**: Nodes should properly clean up resources when shutting down

### Node Composition
- **Single Process**: Multiple nodes can run in a single process for efficiency
- **Component Architecture**: Components can be loaded/unloaded dynamically
- **Resource Sharing**: Shared memory between components in the same process

## Topics and Publisher-Subscriber Pattern

### Topics
Topics are named buses over which nodes exchange messages. The topic name is a unique identifier that allows nodes to send and receive data.

### Publishers
- **Message Creation**: Publishers create and send messages to topics
- **QoS Configuration**: Publishers configure Quality of Service settings
- **Connection Management**: Publishers handle connections to subscribers

### Subscribers
- **Callback Registration**: Subscribers register callbacks to process messages
- **Message Processing**: Subscribers handle incoming messages asynchronously
- **QoS Matching**: Subscribers must match publisher QoS for communication

### Message Types
- **Standard Messages**: Common message types defined in `std_msgs`, `geometry_msgs`, etc.
- **Custom Messages**: User-defined message types created in `.msg` files
- **Message Generation**: Messages automatically generated for multiple languages

### Topic Communication Characteristics
- **Anonymous**: Publishers and subscribers don't know about each other
- **Many-to-Many**: Multiple publishers can send to a topic, multiple subscribers can receive
- **Asynchronous**: Publishers and subscribers don't need to run simultaneously
- **Loose Coupling**: Publishers and subscribers are decoupled in time and space

## Services and Client-Server Pattern

### Services
Services provide synchronous request-response communication between nodes. A service has a specific name and a defined interface.

### Service Servers
- **Request Handling**: Service servers process incoming requests
- **Response Generation**: Servers generate appropriate responses
- **Blocking Operations**: Service calls block until response is received

### Service Clients
- **Request Sending**: Clients send requests to service servers
- **Response Waiting**: Clients wait for and process responses
- **Error Handling**: Clients handle service errors and timeouts

### Service Characteristics
- **Synchronous**: Client waits for response before continuing
- **One-to-One**: One request generates one response
- **Stateless**: Each request is independent of others
- **Reliable**: Request-response is guaranteed (with appropriate QoS)

## Actions

### Action Architecture
Actions are designed for long-running tasks and consist of three parts:
- **Goal**: Request to perform an action
- **Feedback**: Periodic updates on action progress
- **Result**: Final outcome of the action

### Action Servers
- **Goal Acceptance**: Decide whether to accept or reject incoming goals
- **Execution Management**: Execute the action while providing feedback
- **Result Generation**: Provide final result when action completes
- **Preemption Handling**: Cancel ongoing actions when requested

### Action Clients
- **Goal Sending**: Send goals to action servers
- **Feedback Monitoring**: Monitor progress through feedback
- **Result Waiting**: Wait for and process final results
- **Goal Canceling**: Cancel goals if needed

### Action Characteristics
- **Long-Running**: Designed for tasks that take significant time
- **Progress Updates**: Provide feedback during execution
- **Cancelation Support**: Allow goals to be canceled
- **Preemption**: Support for interrupting ongoing actions

## Communication Pattern Selection

### When to Use Topics
- **Sensor Data**: Publishing sensor readings (camera images, LIDAR scans)
- **Robot State**: Broadcasting robot state information
- **Event Notification**: Broadcasting events to multiple subscribers
- **Real-time Requirements**: When low latency is important

### When to Use Services
- **Simple Queries**: Requesting specific information
- **Configuration**: Setting parameters or configuration
- **Brief Operations**: Tasks that complete quickly
- **State Requests**: Getting current state of a system

### When to Use Actions
- **Long-Running Tasks**: Navigation, manipulation, calibration
- **Progress Monitoring**: Tasks where users need progress updates
- **Cancelable Operations**: Tasks that might need to be interrupted
- **Complex Results**: Operations that return complex result data

## Quality of Service (QoS) Considerations

### Reliability Policy
- **Reliable**: All messages are guaranteed to be delivered
- **Best Effort**: Messages may be lost, but delivery is faster

### Durability Policy
- **Transient Local**: Late-joining subscribers receive last message
- **Volatile**: No historical data provided to new subscribers

### History Policy
- **Keep Last**: Maintain only the most recent messages
- **Keep All**: Maintain all messages (use with care)

## Learning Objectives

After completing this section, you should be able to:
- Create and manage ROS 2 nodes in both C++ and Python
- Implement publisher-subscriber communication patterns
- Design and use service-based communication
- Implement action-based communication for long-running tasks
- Choose appropriate communication patterns for different scenarios
- Configure Quality of Service settings for different use cases

## Code Examples

Refer to the following code examples in the textbook repository:
- `docs/static/code-examples/ros2-examples/simple_publisher.py` - Basic publisher node example
- `docs/static/code-examples/ros2-examples/simple_subscriber.py` - Basic subscriber node example

## Exercises

1. Implement a simple publisher-subscriber pair that exchanges custom messages
2. Create a service server that performs a mathematical calculation and a client that uses it
3. Design an action server for a robot navigation task with progress feedback
4. Compare the performance characteristics of different communication patterns for a specific use case
5. Run and modify the provided ROS 2 examples to understand publisher-subscriber communication