import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import {ThemeClassNames} from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

// Simple textbook layout component
export default function TextbookLayout({
  children,
}: {
  children: ReactNode;
}): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();

  return (
    <div className={clsx('textbook-content')}>
      <main className={clsx('container')}>
        <div className="row">
          <div className="col">
            <div className="textbook-content">
              {children}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}