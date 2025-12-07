import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import {useDocsSidebar} from '@docusaurus/theme-common/internal';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {translate} from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import DocPageLayout from '@theme/DocPage/Layout';
import type {Props} from '@theme/DocPage/Type';

// Simple textbook layout component that wraps the standard DocPageLayout
export default function TextbookLayout({
  children,
  ...props
}: {
  children: ReactNode;
} & Omit<Props, 'children'>): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  const sidebar = useDocsSidebar();

  return (
    <DocPageLayout {...props}>
      <div className={clsx(ThemeClassNames.docs.docWrapper, 'textbook-content')}>
        <main className={clsx('container', ThemeClassNames.docs.docMain)}>
          <div className="row">
            <div className="col">
              <div className="textbook-content">
                {children}
              </div>
            </div>
          </div>
        </main>
      </div>
    </DocPageLayout>
  );
}