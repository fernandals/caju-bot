"""rag/parser.py: animation steps dropped, consecutive slides grouped."""

from rag.models import Section, SemanticBlock
from rag.parser import (
    SLIDE_GROUP_TOKENS,
    drop_slide_builds,
    group_slides,
)
from rag.splitter import token_length


def slide(page: int, title: str, body: str, chapter: str | None = None) -> Section:
    return Section(
        id=str(page),
        title=title,
        page_start=page,
        page_end=page,
        start_offset=page * 1000,
        end_offset=page * 1000 + len(body),
        content=body,
        blocks=[
            SemanticBlock(
                content=body,
                page_start=page,
                page_end=page,
                start_offset=page * 1000,
                end_offset=page * 1000 + len(body),
            )
        ]
        if body
        else [],
        chapter_number=chapter,
    )


LONG = "uma árvore AVL é uma árvore binária de busca balanceada pela altura"


class TestDropSlideBuilds:
    def test_animation_step_is_dropped(self):
        step = slide(1, "Inserção", "30\n20")
        full = slide(2, "Inserção", "30\n20\n70")

        assert drop_slide_builds([step, full]) == [full]

    def test_different_slides_are_kept(self):
        slides = [slide(1, "AVL", LONG), slide(2, "Rotações", "rotação simples à direita")]

        assert drop_slide_builds(slides) == slides

    def test_last_slide_is_always_kept(self):
        last = slide(1, "Fim", "")

        assert drop_slide_builds([last]) == [last]


class TestGroupSlides:
    def test_small_slides_are_grouped_with_their_titles(self):
        groups = group_slides([slide(1, "AVL", LONG), slide(2, "Rotações", LONG)])

        assert len(groups) == 1
        assert (groups[0].page_start, groups[0].page_end) == (1, 2)
        assert groups[0].title == "AVL"
        assert [b.content for b in groups[0].blocks] == ["AVL", LONG, "Rotações", LONG]

    def test_a_divider_starts_a_new_group(self):
        groups = group_slides(
            [slide(1, "Heap", LONG), slide(2, "Árvores AVL", ""), slide(3, "Definição", LONG)]
        )

        assert [(g.page_start, g.page_end) for g in groups] == [(1, 1), (2, 3)]

    def test_groups_stay_under_the_budget(self):
        body = " ".join(["palavra"] * (SLIDE_GROUP_TOKENS // 3))
        groups = group_slides([slide(p, f"S{p}", body) for p in range(1, 8)])

        assert len(groups) > 1
        assert all(token_length(g.content) <= SLIDE_GROUP_TOKENS for g in groups)

    def test_a_chapter_change_starts_a_new_group(self):
        groups = group_slides([slide(1, "A", LONG, "1"), slide(2, "B", LONG, "2")])

        assert len(groups) == 2

    def test_single_slide_is_unchanged(self):
        only = slide(1, "AVL", LONG)

        assert group_slides([only]) == [only]
