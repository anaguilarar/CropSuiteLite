import pandas as pd

print(">>> main.py LOADED!")

def define_env(env):
    print(">>> define_env LOADED!")

    @env.macro
    def datatable(csv_path):
        print(">>> datatable MACRO CALLED!")
        df = pd.read_csv(csv_path, encoding="utf-8")
        return df.to_html(
            index=False,
            classes="display compact",
            table_id="myTable"
        )