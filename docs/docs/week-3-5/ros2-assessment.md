---
sidebar_position: 5
title: "ROS 2 Assessment: Fundamentals and Implementation"
---

# ROS 2 Assessment: Fundamentals and Implementation

This assessment evaluates your understanding of ROS 2 fundamentals, including architecture, communication patterns, package development, and launch systems.

## Assessment Overview

### Learning Objectives Tested
- Understanding of ROS 2 architecture and core concepts
- Proficiency in creating and managing ROS 2 packages
- Knowledge of communication patterns (topics, services, actions)
- Ability to configure and use launch files and parameters
- Practical implementation skills in Python

## Part 1: Theoretical Questions (40 points)

### Question 1: Architecture (10 points)
Explain the differences between ROS 1 and ROS 2 architectures. In your answer, address:
1. The underlying communication middleware
2. Support for real-time systems
3. Multi-robot system capabilities
4. Quality of Service (QoS) features

### Question 2: Communication Patterns (10 points)
Compare and contrast the three main communication patterns in ROS 2:
1. Publisher-Subscriber (Pub/Sub)
2. Client-Service (C/S)
3. Actions

For each pattern, describe:
- When to use it
- Advantages and disadvantages
- An example use case

### Question 3: Quality of Service (10 points)
Explain Quality of Service (QoS) profiles in ROS 2. Your answer should include:
1. At least 3 QoS policies
2. How QoS affects communication reliability
3. An example scenario where QoS configuration is critical

### Question 4: Launch Systems (10 points)
Compare XML and Python launch files in ROS 2:
1. Advantages of each approach
2. When to use each type
3. How to pass parameters to nodes through launch files

## Part 2: Practical Implementation (60 points)

### Exercise 1: Basic Publisher-Subscriber (20 points)

Create a ROS 2 package called `sensor_simulation` that includes:
1. A publisher node that publishes temperature readings (using `std_msgs/Float64`) to the topic `/temperature`
2. A subscriber node that subscribes to `/temperature` and logs the values
3. A launch file that starts both nodes
4. Parameters to configure the publishing rate and temperature range

**Requirements:**
- Use Python for implementation
- Include proper error handling
- Use appropriate QoS settings
- Add comments explaining your code

### Exercise 2: Service Implementation (20 points)

Extend the `sensor_simulation` package to include:
1. A service server that provides temperature statistics (min, max, average) from a stored history
2. A service client that calls the server and displays the results
3. The service should use a custom message type with appropriate fields

**Requirements:**
- Create a custom service definition file
- Implement proper request/response handling
- Include timeout handling in the client
- Add parameter configuration for the history buffer size

### Exercise 3: Action Implementation (20 points)

Add to the `sensor_simulation` package:
1. An action server that simulates a temperature calibration process
2. An action client that sends calibration goals and monitors progress
3. The action should provide feedback during the calibration process

**Requirements:**
- Define a custom action message with appropriate goal, feedback, and result fields
- Implement proper goal handling with preemption support
- Provide meaningful feedback during execution
- Include error handling for failed calibration attempts

## Part 3: Advanced Configuration (30 points)

### Exercise 4: Launch File Complexity (15 points)

Create a comprehensive launch file that:
1. Starts the publisher, subscriber, service, and action nodes
2. Uses launch arguments to configure the system
3. Includes parameter files for different configurations (simulation vs. real hardware)
4. Uses conditional launching based on arguments
5. Groups related nodes appropriately

### Exercise 5: Parameter Management (15 points)

Implement a parameter management system that:
1. Uses YAML parameter files for configuration
2. Implements parameter callbacks to respond to runtime changes
3. Provides a way to save current parameters to a file
4. Demonstrates parameter validation

## Evaluation Criteria

### Theoretical Questions
- **Completeness**: All required aspects addressed
- **Accuracy**: Correct technical information
- **Clarity**: Clear and well-structured explanations

### Practical Implementation
- **Functionality**: Code works as specified
- **Code Quality**: Proper structure, comments, error handling
- **ROS 2 Best Practices**: Proper use of ROS 2 concepts and patterns
- **Documentation**: Clear README and inline comments

### Advanced Configuration
- **Complexity**: Appropriate use of advanced features
- **Flexibility**: Configurable and reusable components
- **Robustness**: Handles edge cases and errors appropriately

## Submission Requirements

1. **Source Code**: Complete package with all nodes, launch files, and message definitions
2. **Documentation**: README file explaining the package structure and usage
3. **Configuration Files**: All launch and parameter files
4. **Test Results**: Output demonstrating the system working correctly
5. **Answers**: Written answers to theoretical questions

## Grading Rubric

- **Theoretical Questions**: 40% (10 points each)
- **Practical Implementation**: 45% (15% each exercise)
- **Advanced Configuration**: 15% (7.5% each exercise)

## Learning Objectives Assessment

This assessment verifies your ability to:
- Design and implement ROS 2 systems with appropriate communication patterns
- Create maintainable and configurable ROS 2 packages
- Apply ROS 2 concepts to solve practical robotics problems
- Use launch files and parameters for system configuration

## Code Examples Reference

Refer to the following code examples in the textbook repository as references for this assessment:
- `docs/static/code-examples/ros2-examples/simple_publisher.py` - Basic publisher node example
- `docs/static/code-examples/ros2-examples/simple_subscriber.py` - Basic subscriber node example

## Resources Allowed

- ROS 2 documentation
- Online resources and tutorials
- Your own notes and previous code
- ROS 2 community resources

## Time Limit

Complete this assessment within 2 weeks of assignment. Plan your time to allow for testing and debugging.

## Submission Instructions

Submit your assessment as a compressed archive containing:
1. The complete ROS 2 package
2. A PDF document with your theoretical answers
3. A README file with instructions for building and running your code