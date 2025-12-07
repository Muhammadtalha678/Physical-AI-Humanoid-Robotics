# Data Model: Physical AI & Humanoid Robotics Textbook

## Textbook Chapter
- **name**: String (e.g., "Week 1-2: Introduction to Physical AI")
- **description**: String (brief overview of chapter content)
- **learningObjectives**: Array of String (specific learning objectives)
- **contentSections**: Array of ContentSection (organized sections within the chapter)
- **codeExamples**: Array of CodeExample (related code examples)
- **exercises**: Array of Exercise (practice problems)
- **prerequisites**: Array of String (required prior knowledge)

## ContentSection
- **title**: String (section title)
- **type**: Enum (text, code, diagram, video, exercise)
- **content**: String (markdown content)
- **order**: Number (sequence within chapter)

## CodeExample
- **title**: String (descriptive name)
- **language**: String (e.g., "Python", "ROS 2", "Isaac Sim")
- **code**: String (actual code snippet)
- **description**: String (explanation of what the code does)
- **relatedSections**: Array of String (sections where this example is referenced)

## Exercise
- **title**: String (exercise name)
- **description**: String (detailed instructions)
- **difficulty**: Enum (beginner, intermediate, advanced)
- **solution**: String (reference to solution or solution itself)
- **relatedChapter**: String (chapter this exercise belongs to)

## Assessment
- **title**: String (assessment name)
- **type**: Enum (project, simulation, pipeline, capstone)
- **description**: String (detailed requirements)
- **requirements**: Array of String (specific deliverables)
- **evaluationCriteria**: Array of String (how it will be assessed)

## CurriculumModule
- **name**: String (e.g., "ROS 2 Fundamentals")
- **weeks**: Array of Number (week numbers covered)
- **chapters**: Array of TextbookChapter (chapters in this module)
- **learningObjectives**: Array of String (module-level objectives)
- **prerequisites**: Array of String (required prior modules/chapters)

## PlatformSetup
- **name**: String (e.g., "Digital Twin workstation")
- **description**: String (details about the setup)
- **requirements**: Array of String (hardware/software requirements)
- **setupInstructions**: String (step-by-step instructions)
- **compatibility**: Array of String (compatible systems/tools)