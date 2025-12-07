# ADR-0001: Textbook Curriculum Structure

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-07
- **Feature:** 001-book-master-plan
- **Context:** The Physical AI & Humanoid Robotics textbook requires a structured curriculum to effectively guide industry engineers through complex topics in physical AI, robotics, and related technologies. The curriculum must balance comprehensive coverage with logical progression and prerequisite knowledge.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

**Curriculum Organization**: The textbook curriculum will follow a 13-week structured approach with content organized by weeks:
- Weeks 1-2: Introduction to Physical AI
- Weeks 3-5: ROS 2 Fundamentals
- Weeks 6-7: Robot Simulation with Gazebo
- Weeks 8-10: NVIDIA Isaac Platform
- Weeks 11-12: Humanoid Robot Development
- Week 13: Conversational Robotics

**Content Structure**: Each week-based section will include learning objectives, theoretical foundations, practical examples, code samples in Python/ROS 2/Isaac Sim, and assessment materials.

## Consequences

### Positive

- Clear learning progression with appropriate prerequisite knowledge
- Alignment with the specified 13-week training program structure
- Sequential learning path that builds on previous concepts
- Easy navigation and reference for users at different skill levels
- Comprehensive coverage of all required topics
- Supports both self-paced and instructor-led learning

### Negative

- Rigid structure may not accommodate all learning styles
- Fixed 13-week timeline may not suit all learners' schedules
- Dependencies between weeks could create barriers for non-linear learning
- Requires careful coordination to maintain progression across all topics

## Alternatives Considered

**Topic-based Structure**: Organize content by major topics (Physical AI, ROS 2, Simulation, etc.) rather than weeks. This was rejected because it didn't align with the specified 13-week curriculum requirement and would make it harder to track progress in a training program context.

**Project-based Learning**: Structure content around major projects that integrate multiple concepts. This was considered but rejected as the curriculum requires comprehensive coverage of distinct topics across the 13 weeks, with each week building appropriate foundational knowledge.

**Modular Structure**: Independent modules that could be taken in any order. This was rejected because the content has significant interdependencies and sequential learning is essential for understanding complex robotics concepts.

## References

- Feature Spec: specs/001-book-master-plan/spec.md
- Implementation Plan: specs/001-book-master-plan/plan.md
- Related ADRs: None
- Evaluator Evidence: specs/001-book-master-plan/research.md
