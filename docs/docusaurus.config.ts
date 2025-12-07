import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'A comprehensive guide to Physical AI and Humanoid Robotics for industry engineers',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // ----------- IMPORTANT GITHUB PAGES SETTINGS -----------
  url: 'https://muhammadtalha678.github.io',
  baseUrl: '/Physical-AI-Humanoid-Robotics/',

  organizationName: 'Muhammadtalha678',
  projectName: 'Physical-AI-Humanoid-Robotics',

  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/Muhammadtalha678/Physical-AI-Humanoid-Robotics/edit/main/docs/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {type: ['rss', 'atom'], xslt: true},
          editUrl: 'https://github.com/Muhammadtalha678/Physical-AI-Humanoid-Robotics/edit/main/docs/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      items: [
        {type: 'docSidebar', sidebarId: 'tutorialSidebar', label: 'Textbook', position: 'left'},
        {to: '/search', label: 'Search', position: 'left'},
      ],
    },

    algolia: {
      appId: 'YOUR_APP_ID',
      apiKey: 'YOUR_SEARCH_API_KEY',
      indexName: 'physical-ai-humanoid-robotics',
      contextualSearch: true,
      searchPagePath: 'search',
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [{label: 'Introduction', to: '/docs/intro'}],
        },
        {
          title: 'Resources',
          items: [
            {label: 'ROS 2 Documentation', href: 'https://docs.ros.org/en/humble/'},
            {label: 'NVIDIA Isaac Sim', href: 'https://docs.omniverse.nvidia.com/isaacsim/latest/what_is_isaac_sim.html'},
            {label: 'Gazebo Simulator', href: 'https://gazebosim.org/'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'GitHub', href: 'https://github.com/Muhammadtalha678/Physical_AI_Humanoid_Robotics'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
