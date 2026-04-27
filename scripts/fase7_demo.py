from __future__ import annotations


def main() -> None:
    from monai_pipeline.apps import demo as app

    if app.gr is not None:
        app.demo.launch(share=False)


if __name__ == "__main__":
    main()
