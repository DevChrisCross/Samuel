from setuptools import setup

setup(name='samuel',
      version='0.9.2',
      description='Sentiment Analyzing Machine that can Understand Enumerated Languages',
      long_description="""\
          Sentiment Analyzing Machine that can Understand Enumerated Languages (SAMUEL)
          is a NLP library and api directed towards analyzing documents and its sentiments,
          topics, and summary. SAMUEL requires Python 3.0 or higher""",
      url='',
      author='Dennis De Leon, Christian Noel Molina, Patrick Dale Rugayan, Eric Sumalinog',
      author_email='',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.0',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Human Machine Interfaces',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Text Processing',
          'Topic :: Text Processing :: Filters',
          'Topic :: Text Processing :: General',
          'Topic :: Text Processing :: Indexing',
          'Topic :: Text Processing :: Linguistic',
      ],
      keywords=[
          'NLP', 'CL', 'natural language processing', 'sentiment analysis',
          'computational linguistics', 'parsing', 'tagging', 'summarization',
          'tokenizing', 'syntax', 'linguistics', 'language', 'normalization',
          'natural language', 'text analytics', 'topic modelling', 'api'
      ],
      license='MIT',
      packages=['samuel'],
      install_requires=[
          'pyenchant',
          'nltk',
          'spacy',
          'numpy',
          'gensim',
          'pyLDAvis',
          'googletrans',
          'pytest-runner'
      ],
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      entry_points={
        'console_scripts': ['samuel=samuel.command_line:main'],
      }
)
