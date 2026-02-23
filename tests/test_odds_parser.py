"""オッズテーブル解析ロジックのテスト

boatrace.jp の odds3t / odds2tf / odds2kt ページの HTML テーブル構造を
模した HTML をパースし、正しく組番とオッズが抽出されることを検証する。
"""

import pytest

# bs4 がインストールされていない環境ではスキップ
bs4 = pytest.importorskip("bs4")
from bs4 import BeautifulSoup

from src.scraper.race_data import (
    _parse_odds_table_positional,
    _parse_odds_fallback_regex,
)


def _build_3t_html() -> str:
    """3連単テーブルのモック HTML を生成する。

    構造:
    - div.contentsFrame1_inner > table
    - 6 つの tbody（1着 = 1号艇〜6号艇）
    - 各 tbody に 5 行の tr（2着候補）
    - 各 tr に 4 つの td.oddsPoint（3着候補）
    - 合計 120 セル

    オッズ値は "1着*100 + 2着*10 + 3着" の float で埋める
    （例: 1-2-3 → 123.0）ので検証しやすい。
    """
    boats = [1, 2, 3, 4, 5, 6]
    rows_html = []

    for first in boats:
        second_candidates = [b for b in boats if b != first]
        tbody_rows = []
        for second in second_candidates:
            third_candidates = [b for b in boats if b != first and b != second]
            cells = []
            for third in third_candidates:
                odds_val = float(first * 100 + second * 10 + third)
                cells.append(f'<td class="oddsPoint">{odds_val}</td>')
            tbody_rows.append("<tr>" + "".join(cells) + "</tr>")
        rows_html.append("<tbody>" + "".join(tbody_rows) + "</tbody>")

    table_html = "<table>" + "".join(rows_html) + "</table>"
    return (
        '<html><body><div class="contentsFrame1_inner">'
        + table_html
        + "</div></body></html>"
    )


def _build_2tf_html() -> str:
    """2連単テーブルのモック HTML を生成する。

    構造:
    - 6 つの tbody（1着 = 1号艇〜6号艇）
    - 各 tbody に 5 行の tr（2着候補）
    - 各 tr に 1 つの td.oddsPoint
    - 合計 30 セル

    オッズ値は "1着*10 + 2着" の float。
    """
    boats = [1, 2, 3, 4, 5, 6]
    rows_html = []

    for first in boats:
        second_candidates = [b for b in boats if b != first]
        tbody_rows = []
        for second in second_candidates:
            odds_val = float(first * 10 + second)
            tbody_rows.append(
                f'<tr><td class="oddsPoint">{odds_val}</td></tr>'
            )
        rows_html.append("<tbody>" + "".join(tbody_rows) + "</tbody>")

    table_html = "<table>" + "".join(rows_html) + "</table>"
    return (
        '<html><body><div class="contentsFrame1_inner">'
        + table_html
        + "</div></body></html>"
    )


def _build_2kt_html() -> str:
    """2連複テーブルのモック HTML を生成する。

    構造:
    - 5 つの tbody（軸艇番 1〜5）
    - 各 tbody に (6 - 軸艇番) 行の tr
    - 各 tr に 1 つの td.oddsPoint
    - 合計 15 セル

    オッズ値は "小番号*10 + 大番号" の float。
    """
    boats = [1, 2, 3, 4, 5, 6]
    rows_html = []

    for boat_a in boats[:5]:  # 1〜5
        partner_candidates = [b for b in boats if b > boat_a]
        tbody_rows = []
        for boat_b in partner_candidates:
            odds_val = float(boat_a * 10 + boat_b)
            tbody_rows.append(
                f'<tr><td class="oddsPoint">{odds_val}</td></tr>'
            )
        rows_html.append("<tbody>" + "".join(tbody_rows) + "</tbody>")

    table_html = "<table>" + "".join(rows_html) + "</table>"
    return (
        '<html><body><div class="contentsFrame1_inner">'
        + table_html
        + "</div></body></html>"
    )


class TestParseOdds3t:
    """3連単テーブルの位置ベース解析テスト"""

    def test_returns_120_combos(self):
        html = _build_3t_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "3t"
        )
        assert len(result) == 120

    def test_combo_format(self):
        html = _build_3t_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "3t"
        )
        for combo in result:
            parts = combo.split("-")
            assert len(parts) == 3
            assert all(p in "123456" for p in parts)
            # 1着・2着・3着は全て異なる
            assert len(set(parts)) == 3

    def test_specific_combos(self):
        html = _build_3t_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "3t"
        )
        # 1-2-3 → 123.0
        assert result["1-2-3"] == pytest.approx(123.0)
        # 6-5-4 → 654.0
        assert result["6-5-4"] == pytest.approx(654.0)
        # 3-1-6 → 316.0
        assert result["3-1-6"] == pytest.approx(316.0)
        # 2-4-1 → 241.0
        assert result["2-4-1"] == pytest.approx(241.0)

    def test_no_duplicate_combos(self):
        html = _build_3t_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "3t"
        )
        # キーに重複がないこと（dict なので自動的に保証されるが念のため）
        combos = list(result.keys())
        assert len(combos) == len(set(combos))


class TestParseOdds2tf:
    """2連単テーブルの位置ベース解析テスト"""

    def test_returns_30_combos(self):
        html = _build_2tf_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "2tf"
        )
        assert len(result) == 30

    def test_combo_format(self):
        html = _build_2tf_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "2tf"
        )
        for combo in result:
            parts = combo.split("-")
            assert len(parts) == 2
            assert all(p in "123456" for p in parts)
            assert parts[0] != parts[1]

    def test_specific_combos(self):
        html = _build_2tf_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "2tf"
        )
        # 1-2 → 12.0
        assert result["1-2"] == pytest.approx(12.0)
        # 6-1 → 61.0
        assert result["6-1"] == pytest.approx(61.0)
        # 3-5 → 35.0
        assert result["3-5"] == pytest.approx(35.0)

    def test_order_matters(self):
        """2連単は順番が区別される（1-2 と 2-1 は別）"""
        html = _build_2tf_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "2tf"
        )
        assert "1-2" in result
        assert "2-1" in result
        assert result["1-2"] != result["2-1"]


class TestParseOdds2kt:
    """2連複テーブルの位置ベース解析テスト"""

    def test_returns_15_combos(self):
        html = _build_2kt_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "2kt"
        )
        assert len(result) == 15

    def test_combo_format(self):
        html = _build_2kt_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "2kt"
        )
        for combo in result:
            parts = combo.split("=")
            assert len(parts) == 2
            assert all(p in "123456" for p in parts)
            # 小さい番号 = 大きい番号
            assert int(parts[0]) < int(parts[1])

    def test_specific_combos(self):
        html = _build_2kt_html()
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "2kt"
        )
        # 1=2 → 12.0
        assert result["1=2"] == pytest.approx(12.0)
        # 5=6 → 56.0
        assert result["5=6"] == pytest.approx(56.0)
        # 2=5 → 25.0
        assert result["2=5"] == pytest.approx(25.0)


class TestFallbackSelector:
    """テーブルが見つからない場合のフォールバック動作テスト"""

    def test_empty_html_returns_empty(self):
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.contentsFrame1_inner table", "3t"
        )
        assert result == {}

    def test_fallback_selectors(self):
        """div.contentsFrame1_inner が無い場合、.table1 table にフォールバック"""
        boats = [1, 2, 3, 4, 5, 6]
        rows_html = []
        for first in boats:
            second_candidates = [b for b in boats if b != first]
            tbody_rows = []
            for second in second_candidates:
                odds_val = float(first * 10 + second)
                tbody_rows.append(
                    f'<tr><td class="oddsPoint">{odds_val}</td></tr>'
                )
            rows_html.append("<tbody>" + "".join(tbody_rows) + "</tbody>")
        table_html = "<table>" + "".join(rows_html) + "</table>"
        html = (
            '<html><body><div class="table1">'
            + table_html
            + "</div></body></html>"
        )
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_table_positional(
            soup, "div.nonexistent table", "2tf"
        )
        assert len(result) == 30


class TestFallbackRegex:
    """regex フォールバックのテスト"""

    def test_data_id_pattern(self):
        """td.oddsPoint に data-id 属性がある場合"""
        html = """
        <html><body>
        <td class="oddsPoint" data-id="1-2-3">12.5</td>
        <td class="oddsPoint" data-id="4-5-6">99.9</td>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_fallback_regex(soup, r"\d-\d-\d")
        assert result["1-2-3"] == pytest.approx(12.5)
        assert result["4-5-6"] == pytest.approx(99.9)

    def test_inline_combo_and_odds(self):
        """同じ td 内に組番とオッズがある場合"""
        html = """
        <html><body>
        <table><tr>
            <td>1-2-3 12.5</td>
            <td>4-5-6 99.9</td>
        </tr></table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_fallback_regex(soup, r"\d-\d-\d")
        assert result["1-2-3"] == pytest.approx(12.5)
        assert result["4-5-6"] == pytest.approx(99.9)

    def test_adjacent_cell_odds(self):
        """組番とオッズが隣接する td にある場合"""
        html = """
        <html><body>
        <table>
            <tr><td>1-2-3</td><td>12.5</td></tr>
            <tr><td>4-5-6</td><td>99.9</td></tr>
        </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_fallback_regex(soup, r"\d-\d-\d")
        assert result["1-2-3"] == pytest.approx(12.5)
        assert result["4-5-6"] == pytest.approx(99.9)

    def test_comma_separated_odds(self):
        """オッズ値にカンマがある場合（例: 1,234.5）"""
        html = """
        <html><body>
        <td class="oddsPoint" data-id="1-2-3">1,234.5</td>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_odds_fallback_regex(soup, r"\d-\d-\d")
        assert result["1-2-3"] == pytest.approx(1234.5)
