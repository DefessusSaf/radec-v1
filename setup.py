from setuptools import setup, find_packages

setup(
    name="astro_pipeline",
    version="0.1.0",
    description="Пайплайн обработки FITS-изображений и формирования отчётов",
    author="Saf",
    author_email="sofiasittykova@gmail.com",
    license="MIT",
    package_dir={"": "astro_pipeline/main"},
    packages=find_packages(where="astro_pipeline/main/"),
    py_modules=[
        "monitor",
        "astrometry",
        "config_setting",
        "radec_StarObservation",
        "radec_without_mode",
        "region",
    ],

    include_package_data=True,
    install_requires=[

    ],
    entry_points={
        "console_scripts": [
            "astro=monitor:main",
        ],
    },
)