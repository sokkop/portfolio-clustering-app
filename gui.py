# app_gui.py
# Интеллектуальное приложение для формирования диверсифицированного портфеля
# Графический интерфейс на tkinter (график волатильности только внутри окна)

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

plt.style.use('ggplot')


# ========= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (как в консольной версии) =========

def preprocess_prices(prices_df):
    drop_cols = [
        'VTBR', 'CNRU', 'X5', 'KFBA', 'FIXR', 'RAGR',
        'T', 'LMBZ', 'OZPH', 'DATA', 'HEAD',
        'APRI', 'YDEX', 'PRMD', 'VSEH',
        'ELMT', 'IVAT', 'SVETP', 'MBNK', 'ZAYM', 'LEAS'
    ]
    drop_cols_exist = [c for c in drop_cols if c in prices_df.columns]
    if drop_cols_exist:
        prices_df = prices_df.drop(columns=drop_cols_exist)

    prices_df = prices_df.sort_index()
    prices_df = prices_df.fillna(method='ffill')
    prices_df = prices_df.dropna(how='all')

    # фиксы сплитов / деноминаций
    if 'BELU' in prices_df.columns:
        prices_df.loc["2024-08-22":, "BELU"] *= 10
    if 'GEMA' in prices_df.columns:
        prices_df.loc["2024-02-08":, "GEMA"] *= 10
    if 'GMKN' in prices_df.columns:
        prices_df.loc["2024-04-08":, "GMKN"] *= 100
    if 'TRNFP' in prices_df.columns:
        prices_df.loc[:"2024-02-20", "TRNFP"] /= 100
    if 'URKZ' in prices_df.columns:
        prices_df.loc["2025-08-05":, "URKZ"] *= 100
    if 'KOGK' in prices_df.columns:
        prices_df.loc["2025-08-15":, "KOGK"] *= 100
    if 'PLZL' in prices_df.columns:
        prices_df.loc["2025-03-27":, "PLZL"] *= 10

    return prices_df


def fill_na_smart(col: pd.Series) -> pd.Series:
    first_valid = col.first_valid_index()
    if first_valid is None:
        return col.fillna(0)
    filled = col.copy()
    filled.loc[:first_valid] = filled.loc[:first_valid].fillna(0)
    filled = filled.ffill()
    return filled


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change()
    rets = np.log(1 + rets)
    rets = rets.apply(fill_na_smart)
    return rets


def risk_parity_weights(returns_df: pd.DataFrame) -> pd.Series:
    cov = returns_df.cov()
    inv_vol = 1 / np.sqrt(np.diag(cov))
    weights = inv_vol / inv_vol.sum()
    return pd.Series(weights, index=returns_df.columns)


def compute_sharpe(returns_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(returns_df.mean() / returns_df.std(), columns=['Sharpe'])


def portf_sharpe_from_prices(portfolio, prices_df: pd.DataFrame) -> float:
    portf_val = np.zeros(len(prices_df))
    for stock in portfolio:
        portf_val += prices_df[stock]
    portf_returns = np.log(1 + pd.Series(portf_val).pct_change())
    return portf_returns[1:].mean() / portf_returns[1:].std()


def show_portfolio(portfolio_tickers, name, returns_df, text_widget=None):
    """
    Показываем только таблицу портфеля в текстовом поле (без графика структуры).
    """
    portf_df = returns_df[portfolio_tickers]
    weights = risk_parity_weights(portf_df)

    portfolio_final = pd.DataFrame({
        'Ticker': weights.index,
        'Weight (%)': (weights.values * 100).round(2),
        'Mean Return (%)': (portf_df.mean().values * 100).round(3),
        'Volatility (%)': (portf_df.std().values * 100).round(3),
        'Sharpe Ratio': (portf_df.mean() / portf_df.std()).round(3)
    }).sort_values('Weight (%)', ascending=False)

    header = f"\n{'=' * 60}\nИтоговый портфель — {name}\n{'=' * 60}\n"
    body = portfolio_final.to_string(index=False) + "\n"

    if text_widget is not None:
        text_widget.insert(tk.END, header)
        text_widget.insert(tk.END, body)
        text_widget.see(tk.END)
    else:
        print(header)
        print(body)

    # График структуры портфеля убран, чтобы остался только график волатильности внутри окна

    return portfolio_final


# ========= МЕТОДЫ КЛАСТЕРИЗАЦИИ =========

def method_kmeans(returns_train, returns_test, prices_test, clusters_n=10):
    sharpe_ratio = compute_sharpe(returns_train)
    X = returns_train.T
    labels = KMeans(n_clusters=clusters_n, random_state=42, n_init='auto').fit_predict(X)

    df_groups = pd.DataFrame(labels, index=returns_train.columns, columns=['Group'])
    df_groups = pd.concat([df_groups, sharpe_ratio], axis=1)

    portfolio_kmeans = df_groups.groupby('Group')['Sharpe'].idxmax().tolist()
    sharpe_kmeans = portf_sharpe_from_prices(portfolio_kmeans, prices_test)
    return portfolio_kmeans, sharpe_kmeans


def method_pca_kmeans(returns_train, returns_test, prices_test,
                      n_components=7, clusters_max=30):
    X = returns_train.T.values
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    results = []
    k_values = range(2, clusters_max + 1)
    best_portfolio = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_pca)

        df_groups_pca = pd.DataFrame({
            'Group': labels,
            'Sharpe': (returns_train.mean() / returns_train.std()).values
        }, index=returns_train.columns)

        portfolio_pca = (df_groups_pca.groupby('Group')
                         .apply(lambda g: g['Sharpe'].idxmax())
                         .tolist())

        portf_df = returns_test[portfolio_pca]
        weights_rp = risk_parity_weights(portf_df)
        portf_returns = (portf_df * weights_rp).sum(axis=1)
        sharpe_final = portf_returns.mean() / portf_returns.std()

        results.append((k, sharpe_final))

        if best_portfolio is None or sharpe_final > best_portfolio[1]:
            best_portfolio = (portfolio_pca, sharpe_final)

    portfolio_best, sharpe_best = best_portfolio
    return portfolio_best, sharpe_best


def method_hierarchical(returns_train, returns_test, prices_test,
                        clusters_n=10):
    methods = ['single', 'complete', 'average']
    sharpe_ratio = compute_sharpe(returns_train)

    corr = pd.DataFrame(np.corrcoef(returns_train, rowvar=False),
                        columns=returns_train.columns, index=returns_train.columns)
    D = 1 - corr

    result = {}

    for linkage in methods:
        agg = AgglomerativeClustering(
            n_clusters=clusters_n,
            metric='precomputed',
            linkage=linkage
        )
        groups = agg.fit_predict(D)
        df_groups = pd.DataFrame(groups, index=returns_train.columns, columns=['Group'])
        df_groups = pd.concat([df_groups, sharpe_ratio], axis=1)

        portfolio = df_groups.groupby('Group')['Sharpe'].idxmax().tolist()
        sharpe_score = portf_sharpe_from_prices(portfolio, prices_test)
        result[linkage] = (portfolio, sharpe_score)

    return result


def benchmark_random_portfolio(returns_train, prices_test, n_stocks=30, n_iter=30):
    cols = returns_train.columns
    sum_sharpe = 0
    for i in range(n_iter):
        np.random.seed(i)
        rand_idx = np.random.randint(0, len(cols), n_stocks)
        rand_portf = cols[rand_idx]
        sum_sharpe += portf_sharpe_from_prices(rand_portf, prices_test)
    return sum_sharpe / n_iter


def compute_volatilities(returns_test, portfolios_dict):
    vol_dict = {}

    # рынок
    weights_market = np.ones(len(returns_test.columns)) / len(returns_test.columns)
    market_vol = sqrt(np.dot(weights_market.T, np.dot(returns_test.cov() * 252, weights_market)))
    vol_dict['Market'] = market_vol

    # случайный
    np.random.seed(0)
    rand_idx = np.random.randint(0, len(returns_test.columns), 30)
    rand_portf = returns_test.columns[rand_idx]
    portf_df_rand = returns_test[rand_portf]
    weights_rand = np.ones(30) / 30
    rand_vol = sqrt(np.dot(weights_rand.T, np.dot(portf_df_rand.cov() * 252, weights_rand)))
    vol_dict['Random'] = rand_vol

    for name, tickers in portfolios_dict.items():
        portf_df = returns_test[tickers]
        w = np.ones(len(portf_df.columns)) / len(portf_df.columns)
        vol = sqrt(np.dot(w.T, np.dot(portf_df.cov() * 252, w)))
        vol_dict[name] = vol

    vol_df = pd.DataFrame(vol_dict, index=['Annualized Volatility']).T
    return vol_df


# ========= TKINTER GUI =========

class PortfolioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Формирование диверсифицированного портфеля (кластерный анализ)")
        self.geometry("1100x700")

        self.prices_df = None
        self.returns_train = None
        self.returns_test = None
        self.prices_train = None
        self.prices_test = None

        self.canvas_vol = None  # холст для графика волатильности

        self._build_widgets()

    def _build_widgets(self):
        frame_top = ttk.Frame(self)
        frame_top.pack(fill=tk.X, padx=10, pady=10)

        # Кнопка выбора файла
        self.btn_file = ttk.Button(frame_top, text="Выбрать CSV-файл", command=self.choose_file)
        self.btn_file.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Путь к файлу
        self.var_filepath = tk.StringVar()
        entry_file = ttk.Entry(frame_top, textvariable=self.var_filepath, width=80)
        entry_file.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Дата разделения
        ttk.Label(frame_top, text="Дата разделения (ГГГГ-ММ-ДД):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.var_split_date = tk.StringVar(value="2025-01-01")
        entry_date = ttk.Entry(frame_top, textvariable=self.var_split_date, width=20)
        entry_date.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Выбор метода
        ttk.Label(frame_top, text="Метод кластеризации:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.var_method = tk.StringVar()
        self.combo_method = ttk.Combobox(
            frame_top,
            textvariable=self.var_method,
            values=[
                "K-Means",
                "PCA + K-Means",
                "Hierarchical Single",
                "Hierarchical Complete",
                "Hierarchical Average",
                "Все методы"
            ],
            state="readonly",
            width=30
        )
        self.combo_method.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.combo_method.current(5)  # по умолчанию "Все методы"

        # Кнопка запуска
        self.btn_run = ttk.Button(frame_top, text="Запустить анализ", command=self.run_analysis)
        self.btn_run.grid(row=3, column=0, padx=5, pady=10, sticky="w")

        # Основная нижняя часть: текст + график
        frame_bottom = ttk.Frame(self)
        frame_bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Текстовое поле для вывода
        frame_text = ttk.Frame(frame_bottom)
        frame_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.text_output = scrolledtext.ScrolledText(frame_text, wrap=tk.WORD, font=("Consolas", 10))
        self.text_output.pack(fill=tk.BOTH, expand=True)

        # Фрейм для графика волатильности
        self.frame_plot = ttk.Frame(frame_bottom)
        self.frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    def choose_file(self):
        filepath = filedialog.askopenfilename(
            title="Выберите CSV-файл с ценами акций",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.var_filepath.set(filepath)

    def log(self, msg: str):
        self.text_output.insert(tk.END, msg + "\n")
        self.text_output.see(tk.END)
        self.update_idletasks()

    def run_analysis(self):
        self.text_output.delete('1.0', tk.END)  # очистка

        filepath = self.var_filepath.get().strip()
        if not filepath:
            messagebox.showerror("Ошибка", "Сначала выберите CSV-файл.")
            return

        if not os.path.exists(filepath):
            messagebox.showerror("Ошибка", f"Файл не найден:\n{filepath}")
            return

        # Загрузка
        try:
            self.log(f"Загрузка файла: {filepath}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        except Exception as e:
            messagebox.showerror("Ошибка чтения файла", str(e))
            return

        if df.empty or df.shape[1] < 1:
            messagebox.showerror("Ошибка", "Файл не содержит корректные данные о ценах акций.")
            return

        # Предобработка
        df = preprocess_prices(df)
        self.prices_df = df

        self.log(f"Файл загружен. Тикеров: {df.shape[1]}, дней: {df.shape[0]}.")

        # Проверка и разбиение по дате
        split_str = self.var_split_date.get().strip()
        try:
            split_date = pd.to_datetime(split_str)
        except Exception:
            messagebox.showerror("Ошибка", "Некорректный формат даты. Используйте ГГГГ-ММ-ДД.")
            return

        if not (df.index.min() <= split_date <= df.index.max()):
            messagebox.showerror(
                "Ошибка",
                f"Дата {split_str} вне диапазона данных: {df.index.min().date()} ... {df.index.max().date()}"
            )
            return

        self.prices_train = df.loc[:split_date]
        self.prices_test = df.loc[split_date:]

        if self.prices_test.empty:
            messagebox.showerror("Ошибка", "После даты разделения нет данных для тестовой выборки.")
            return

        self.log(f"Train: {self.prices_train.shape}, Test: {self.prices_test.shape}")

        # Доходности
        self.returns_train = compute_log_returns(self.prices_train)
        self.returns_test = compute_log_returns(self.prices_test)

        # Выбор метода
        method = self.var_method.get()
        self.log(f"Выбран метод кластеризации: {method}")

        # Бенчмарк: случайные портфели
        sharpe_random = benchmark_random_portfolio(self.returns_train, self.prices_test)
        self.log(f"Средний Sharpe для случайных портфелей: {sharpe_random:.4f}")

        sharpe_summary = [("Random", sharpe_random)]
        portfolios_all = {}

        # Запуск выбранных методов
        try:
            if method in ("K-Means", "Все методы"):
                port_km, sh_km = method_kmeans(self.returns_train, self.returns_test, self.prices_test)
                show_portfolio(port_km, "KMeans", self.returns_test, self.text_output)
                sharpe_summary.append(("KMeans", sh_km))
                portfolios_all["KMeans"] = port_km

            if method in ("PCA + K-Means", "Все методы"):
                port_pca, sh_pca = method_pca_kmeans(self.returns_train, self.returns_test, self.prices_test)
                show_portfolio(port_pca, "PCA+KMeans", self.returns_test, self.text_output)
                sharpe_summary.append(("PCA+KMeans", sh_pca))
                portfolios_all["PCA+KMeans"] = port_pca

            if method in ("Hierarchical Single", "Hierarchical Complete", "Hierarchical Average", "Все методы"):
                hier_res = method_hierarchical(self.returns_train, self.returns_test, self.prices_test)

                if method in ("Hierarchical Single", "Все методы"):
                    p_single, sh_single = hier_res['single']
                    show_portfolio(p_single, "Hierarchical Single", self.returns_test, self.text_output)
                    sharpe_summary.append(("HierSingle", sh_single))
                    portfolios_all["Single linkage"] = p_single

                if method in ("Hierarchical Complete", "Все методы"):
                    p_complete, sh_complete = hier_res['complete']
                    show_portfolio(p_complete, "Hierarchical Complete", self.returns_test, self.text_output)
                    sharpe_summary.append(("HierComp", sh_complete))
                    portfolios_all["Complete linkage"] = p_complete

                if method in ("Hierarchical Average", "Все методы"):
                    p_avg, sh_avg = hier_res['average']
                    show_portfolio(p_avg, "Hierarchical Average", self.returns_test, self.text_output)
                    sharpe_summary.append(("HierAvg", sh_avg))
                    portfolios_all["Average linkage"] = p_avg

        except Exception as e:
            messagebox.showerror("Ошибка при расчётах", str(e))
            return

        # Сводка Sharpe
        summary_df = pd.DataFrame(sharpe_summary, columns=['Method', 'Sharpe']).sort_values('Sharpe', ascending=False)
        self.log("\nСводка Sharpe Ratio по методам:")
        self.log(summary_df.to_string(index=False))

        best_method = summary_df.iloc[0]
        self.log(f"\nЛучший метод по Sharpe Ratio: {best_method['Method']} ({best_method['Sharpe']:.4f})")

        # Волатильности
        vol_df = compute_volatilities(self.returns_test, portfolios_all)
        self.log("\nГодовая волатильность (Annualized Volatility):")
        self.log(vol_df.round(6).to_string())

        # --- РИСУЕМ ТОЛЬКО ГРАФИК ВОЛАТИЛЬНОСТИ ВНУТРИ ОКНА TKINTER ---

        # Если уже есть старый график, удаляем его
        if self.canvas_vol is not None:
            self.canvas_vol.get_tk_widget().destroy()
            self.canvas_vol = None

        # Создаём Figure и ось
        fig = Figure(figsize=(5.5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        vol_df.plot(kind='bar', ax=ax)
        ax.set_ylabel('Annualized Volatility')
        ax.set_title('Годовая волатильность')
        ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()

        # Встраиваем в Tkinter
        self.canvas_vol = FigureCanvasTkAgg(fig, master=self.frame_plot)
        self.canvas_vol.draw()
        self.canvas_vol.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.log("\nАнализ завершён.\n")


if __name__ == "__main__":
    app = PortfolioApp()
    app.mainloop()