import os
import sys
from manim import *

config.tex_tracker = None
config.use_tex = False
config.tex_program = "notepad" 

class VersionCompareScene(Scene):
    def construct(self):
        title = Text("Advanced Face Detection Infrastructure", font_size=26, color=BLUE)
        subtitle = Text("Algorithmic Paradigm Shift: Haar-Cascade vs RetinaFace FPN", font_size=15, color=GRAY)
        author = Text("Technical Evolution Analysis | Team T09", font_size=13, color=LIGHT_GRAY)
        
        subtitle.next_to(title, DOWN, buff=0.25)
        author.next_to(subtitle, DOWN, buff=0.4)
        
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.2)
        self.play(FadeIn(author), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(author), run_time=1.0)

        left_title = Text("Version 1: Haar Classifier Pipeline", font_size=18, color=RED).shift(LEFT * 3.5 + UP * 3.2)
        right_title = Text("Version 2: RetinaFace Multi-Task FPN", font_size=18, color=GREEN).shift(RIGHT * 3.5 + UP * 3.2)
        divider = Line(UP * 3.5, DOWN * 3.5, color=GRAY)
        self.play(FadeIn(left_title), FadeIn(right_title), Create(divider), run_time=1.0)
        self.wait(0.5)

        grid = Rectangle(width=2.8, height=2.8, grid_xstep=0.35, grid_ystep=0.35, color=WHITE, stroke_width=1.5)
        grid.shift(LEFT * 3.5 + UP * 0.5)
        v1_label = Text("Adaboost Cascaded Scan & Integral Image", font_size=11, color=LIGHT_GRAY).next_to(grid, DOWN, buff=0.3)
        
        v1_cap1 = Text("V1 Mechanism: Operates via hand-crafted Haar-like features", font_size=12, color=RED).to_edge(DOWN, buff=0.8)
        v1_cap2 = Text("and applies dense sliding window cross-correlation on input matrix.", font_size=12, color=RED).next_to(v1_cap1, DOWN, buff=0.1)
        
        self.play(Create(grid), FadeIn(v1_label), run_time=1.0)
        self.play(FadeIn(v1_cap1), FadeIn(v1_cap2), run_time=0.8)
        self.wait(1.0)

        feat_horizontal = VGroup(
            Rectangle(width=0.7, height=0.35, fill_color=WHITE, fill_opacity=0.8, stroke_width=1),
            Rectangle(width=0.7, height=0.35, fill_color=BLACK, fill_opacity=0.9, stroke_width=1)
        ).arrange(RIGHT, buff=0).move_to(grid.get_corner(UL) + RIGHT * 0.35 + DOWN * 0.175)

        feat_vertical = VGroup(
            Rectangle(width=0.35, height=0.7, fill_color=WHITE, fill_opacity=0.8, stroke_width=1),
            Rectangle(width=0.35, height=0.7, fill_color=BLACK, fill_opacity=0.9, stroke_width=1),
            Rectangle(width=0.35, height=0.7, fill_color=WHITE, fill_opacity=0.8, stroke_width=1)
        ).arrange(RIGHT, buff=0).move_to(grid.get_corner(UL) + RIGHT * 1.4 + DOWN * 0.7)

        slide_box = Rectangle(width=1.4, height=1.4, color=RED, stroke_width=3)
        slide_box.move_to(grid.get_corner(UL) + RIGHT * 0.7 + DOWN * 0.7)
        
        self.play(FadeIn(slide_box), FadeIn(feat_horizontal), run_time=0.6)
        self.play(slide_box.animate.shift(RIGHT * 0.35), feat_horizontal.animate.shift(RIGHT * 0.35), run_time=0.5)
        self.play(slide_box.animate.shift(RIGHT * 0.35), feat_horizontal.animate.shift(RIGHT * 0.35), run_time=0.5)
        
        v1_fail1 = Text("Cascade Attenuation: Rigid structural models yield catastrophic failures", font_size=12, color=WHITE).to_edge(DOWN, buff=0.8)
        v1_fail2 = Text("when encountering severe variations such as deep profile angles or shadows.", font_size=12, color=WHITE).next_to(v1_fail1, DOWN, buff=0.1)
        
        self.play(FadeOut(v1_cap1), FadeOut(v1_cap2), run_time=0.4)
        self.play(FadeIn(v1_fail1), FadeIn(v1_fail2), run_time=0.6)
        
        self.play(FadeIn(feat_vertical), run_time=0.4)
        self.play(slide_box.animate.shift(DOWN * 0.35 + LEFT * 0.35), run_time=0.5)
        self.play(slide_box.animate.shift(RIGHT * 0.35), run_time=0.5)
        self.wait(1.5)
        
        self.play(FadeOut(v1_fail1), FadeOut(v1_fail2), run_time=0.5)

        layer3 = Rectangle(width=3.2, height=0.3, color=BLUE_E, fill_opacity=0.4).shift(RIGHT * 3.5 + DOWN * 0.5)
        layer2 = Rectangle(width=2.0, height=0.3, color=BLUE_C, fill_opacity=0.5).next_to(layer3, UP, buff=0.6)
        layer1 = Rectangle(width=1.0, height=0.3, color=BLUE_A, fill_opacity=0.6).next_to(layer2, UP, buff=0.6)
        
        l3_lbl = Text("P3: Low-Level (Small Face Detection)", font_size=9, color=GRAY).next_to(layer3, RIGHT, buff=0.1)
        l2_lbl = Text("P4: Mid-Level", font_size=9, color=GRAY).next_to(layer2, RIGHT, buff=0.1)
        l1_lbl = Text("P5: High-Level (Large Face Detection)", font_size=9, color=GRAY).next_to(layer1, RIGHT, buff=0.1)
        
        v2_label = Text("Feature Pyramid Network & Multi-Scale Semantic Fusion", font_size=11, color=LIGHT_GRAY).next_to(layer3, DOWN, buff=0.4)
        
        v2_cap1 = Text("V2 Architecture: Leverages ResNet backbone with top-down lateral connections.", font_size=12, color=GREEN).to_edge(DOWN, buff=0.8)
        v2_cap2 = Text("Constructs rich feature hierarchies to encapsulate different spatial resolutions.", font_size=12, color=GREEN).next_to(v2_cap1, DOWN, buff=0.1)

        self.play(Create(layer3), FadeIn(v2_label), FadeIn(l3_lbl), run_time=0.8)
        self.play(FadeIn(v2_cap1), FadeIn(v2_cap2), run_time=0.6)
        self.wait(0.8)

        line1 = Line(layer3.get_top(), layer2.get_bottom(), color=YELLOW, stroke_width=2)
        self.play(Create(line1), Create(layer2), FadeIn(l2_lbl), run_time=0.6)
        
        line2 = Line(layer2.get_top(), layer1.get_bottom(), color=YELLOW, stroke_width=2)
        self.play(Create(line2), Create(layer1), FadeIn(l1_lbl), run_time=0.6)
        self.wait(0.5)

        v2_succ1 = Text("Multi-Task Joint Regression: Concurrently outputs high-precision bounding boxes", font_size=12, color=WHITE).to_edge(DOWN, buff=0.8)
        v2_succ2 = Text("along with 5 dense localized facial landmarks for optimal geometric alignment.", font_size=12, color=WHITE).next_to(v2_succ1, DOWN, buff=0.1)
        
        self.play(FadeOut(v2_cap1), FadeOut(v2_cap2), run_time=0.4)
        self.play(FadeIn(v2_succ1), FadeIn(v2_succ2), run_time=0.6)

        det_large = Rectangle(width=1.3, height=1.3, color=YELLOW, stroke_width=2.5).move_to(layer3.get_center())
        det_small = Rectangle(width=0.4, height=0.4, color=YELLOW, stroke_width=2.5).move_to(layer1.get_center())
        
        l_pts = [det_large.get_center() + p for p in [LEFT*0.3+UP*0.2, RIGHT*0.3+UP*0.2, ORIGIN, LEFT*0.2+DOWN*0.3, RIGHT*0.2+DOWN*0.3]]
        s_pts = [det_small.get_center() + p for p in [LEFT*0.08+UP*0.05, RIGHT*0.08+UP*0.05, ORIGIN, LEFT*0.05+DOWN*0.08, RIGHT*0.05+DOWN*0.08]]
        
        landmarks = VGroup()
        for p in l_pts + s_pts:
            landmarks.add(Dot(point=p, radius=0.035, color=ORANGE))

        self.play(Create(det_large), Create(det_small), FadeIn(landmarks), run_time=1.2)
        self.wait(3.5)

        self.play(
            VGroup(grid, slide_box, feat_horizontal, feat_vertical, v1_label).animate.scale(0.8).set_opacity(0),
            VGroup(layer3, layer2, layer1, line1, line2, l3_lbl, l2_lbl, l1_lbl, det_large, det_small, landmarks, v2_label).animate.scale(0.8).set_opacity(0),
            FadeOut(left_title), FadeOut(right_title), FadeOut(divider),
            FadeOut(v2_succ1), FadeOut(v2_succ2),
            run_time=1.2
        )

        step3_title = Text("Step 3: Quantitative Robustness Analysis (15 Evaluation Benchmarks)", font_size=18, color=BLUE)
        step3_title.to_edge(UP, buff=0.5)
        self.play(FadeIn(step3_title), run_time=0.8)

        origin = DOWN * 1.3 + LEFT * 3.5
        x_axis = Line(origin, origin + RIGHT * 7.2, color=WHITE, stroke_width=2)
        y_axis = Line(origin, origin + UP * 4.0, color=WHITE, stroke_width=2)
        
        x_text = Text("Test Dataset Image Index (1 - 15)", font_size=11).next_to(x_axis, DOWN, buff=0.3)
        y_text = Text("mAP Performance Curve (%)", font_size=11).next_to(y_axis, UP, buff=0.2).shift(RIGHT * 0.4)
        
        y_0 = Text("0%", font_size=10).next_to(origin, LEFT, buff=0.15)
        y_50 = Text("50%", font_size=10).next_to(origin + UP * 1.8, LEFT, buff=0.15)
        y_100 = Text("100%", font_size=10).next_to(origin + UP * 3.6, LEFT, buff=0.15)
        
        self.play(Create(x_axis), Create(y_axis), run_time=1.0)
        self.play(FadeIn(x_text), FadeIn(y_text), FadeIn(y_0), FadeIn(y_50), FadeIn(y_100), run_time=0.8)
        self.wait(0.8)

        graph_cap1 = Text("Comparative Validation: Under complex and unconstrained validation sets,", font_size=12, color=LIGHT_GRAY).to_edge(DOWN, buff=0.7)
        graph_cap2 = Text("Haar-Cascade demonstrates significant variance due to hand-crafted threshold bounds.", font_size=12, color=LIGHT_GRAY).next_to(graph_cap1, DOWN, buff=0.1)
        self.play(FadeIn(graph_cap1), FadeIn(graph_cap2), run_time=0.6)

        v1_points = [
            origin + RIGHT * 0.1 + UP * 3.3, origin + RIGHT * 0.6 + UP * 1.1,
            origin + RIGHT * 1.2 + UP * 2.9, origin + RIGHT * 1.8 + UP * 0.5,
            origin + RIGHT * 2.4 + UP * 2.4, origin + RIGHT * 3.0 + UP * 0.9,
            origin + RIGHT * 3.6 + UP * 3.1, origin + RIGHT * 4.2 + UP * 0.4,
            origin + RIGHT * 4.8 + UP * 2.2, origin + RIGHT * 5.4 + UP * 0.7,
            origin + RIGHT * 6.0 + UP * 2.7, origin + RIGHT * 6.6 + UP * 1.3,
            origin + RIGHT * 7.1 + UP * 3.2
        ]
        v1_lines = VGroup()
        v1_dots = VGroup()
        for i in range(len(v1_points)-1):
            v1_lines.add(Line(v1_points[i], v1_points[i+1], color=RED, stroke_width=3))
        for pt in v1_points:
            v1_dots.add(Dot(pt, color=RED, radius=0.05))

        label_v1_line = Text("V1 (Haar): High Variance / Unstable", font_size=11, color=RED).move_to(UP * 0.8 + RIGHT * 1.8)
        self.play(Create(v1_lines), Create(v1_dots), FadeIn(label_v1_line), run_time=2.5)
        self.wait(1.5)

        graph_cap3 = Text("Conversely, RetinaFace retains invariant feature representation architectures,", font_size=12, color=GREEN).to_edge(DOWN, buff=0.7)
        graph_cap4 = Text("establishing 100% stable precision across all complex lighting benchmarks.", font_size=12, color=GREEN).next_to(graph_cap3, DOWN, buff=0.1)
        
        self.play(FadeOut(graph_cap1), FadeOut(graph_cap2), run_time=0.4)
        self.play(FadeIn(graph_cap3), FadeIn(graph_cap4), run_time=0.6)

        v2_start = origin + UP * 3.6
        v2_end = origin + RIGHT * 7.1 + UP * 3.6
        v2_line = Line(v2_start, v2_end, color=GREEN, stroke_width=3.5)
        v2_dot1 = Dot(v2_start, color=GREEN, radius=0.05)
        v2_dot2 = Dot(v2_end, color=GREEN, radius=0.05)

        label_v2_line = Text("V2 (RetinaFace): High Generalization Invariance", font_size=11, color=GREEN).move_to(UP * 2.2 + RIGHT * 1.8)
        self.play(Create(v2_line), Create(v2_dot1), Create(v2_dot2), FadeIn(label_v2_line), run_time=2.0)
        self.wait(3.5)



#Asked AI for help        