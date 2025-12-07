import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the 13-week curriculum
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['week-1-2/intro'],
    },
    {
      type: 'category',
      label: 'Weeks 1-2: Introduction to Physical AI',
      items: [
        'week-1-2/introduction-to-physical-ai',
        'week-1-2/foundations-physical-ai',
        'week-1-2/digital-ai-robots',
        'week-1-2/humanoid-robotics-landscape',
        'week-1-2/sensor-systems'
      ],
    },
    {
      type: 'category',
      label: 'Weeks 3-5: ROS 2 Fundamentals',
      items: [
        'week-3-5/ros2-architecture',
        'week-3-5/ros2-nodes-topics',
        'week-3-5/ros2-packages-python',
        'week-3-5/ros2-launch-files',
        'week-3-5/ros2-assessment'
      ],
    },
    {
      type: 'category',
      label: 'Weeks 6-7: Robot Simulation with Gazebo',
      items: [
        'week-6-7/gazebo-setup',
        'week-6-7/urdf-sdf-formats',
        'week-6-7/gazebo-assessment'
      ],
    },
    {
      type: 'category',
      label: 'Weeks 8-10: NVIDIA Isaac Platform',
      items: [
        'week-8-10/isaac-sdk-sim',
        'week-8-10/ai-perception-manipulation',
        'week-8-10/reinforcement-learning-control',
        'week-8-10/sim-to-real-transfer',
        'week-8-10/isaac-assessment'
      ],
    },
    {
      type: 'category',
      label: 'Weeks 11-12: Humanoid Robot Development',
      items: [
        'week-11-12/humanoid-kinematics-dynamics',
        'week-11-12/bipedal-locomotion',
        'week-11-12/manipulation-grasping',
        'week-11-12/human-robot-interaction'
      ],
    },
    {
      type: 'category',
      label: 'Week 13: Conversational Robotics',
      items: [
        'week-13/gpt-conversational-ai',
        'week-13/speech-recognition',
        'week-13/multi-modal-interaction'
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone-project',
        'platform-setup/digital-twin',
        'platform-setup/edge-kit',
        'platform-setup/cloud-native'
      ],
    }
  ],
};

export default sidebars;
