# Quickstart: Physical AI & Humanoid Robotics Textbook

## Prerequisites
- Node.js (v16 or higher)
- Git
- Basic knowledge of Python (for understanding code examples)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies:**
   ```bash
   cd docs
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```
   This will start the Docusaurus server and open the textbook in your browser at http://localhost:3000

4. **Build for production:**
   ```bash
   npm run build
   ```
   This creates a static build in the `build/` directory

5. **Deploy to GitHub Pages:**
   ```bash
   npm run deploy
   ```
   This deploys the textbook to GitHub Pages based on your configuration

## Adding New Content

1. **Create a new markdown file** in the appropriate week directory under `docs/`
2. **Follow the existing content structure** with appropriate headers and metadata
3. **Update `sidebars.js`** to include your new content in the navigation
4. **Test locally** with `npm start` before committing

## Content Guidelines

- All content should follow the 13-week curriculum structure
- Code examples should be in Python, ROS 2, or Isaac Sim as appropriate
- Include learning objectives at the beginning of each section
- Add exercises and assessments where relevant
- Maintain accessibility standards for all content