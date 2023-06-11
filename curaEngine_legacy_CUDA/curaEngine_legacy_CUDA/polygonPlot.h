#pragma once

#include <Windows.h>
#include "utils/polygon.h"

namespace cura {

	void linePlot(HDC hdc, HPEN hpen, Point pointStart, Point pointEnd)
	{
		SelectObject(hdc, hpen);
		//	(HPEN)SelectObject(hdc, hpen);
		MoveToEx(hdc, pointStart.X, pointStart.Y, NULL);
		LineTo(hdc, pointEnd.X, pointEnd.Y);
	}

	void polygonPlot(HDC hdc, HPEN hpen, PolygonRef out_line) {
		int p0 = out_line.size() - 1;
		for (int p1 = 0; p1 < out_line.size(); ++p1) {
			linePlot(hdc, hpen, out_line[p0], out_line[p1]);
			p0 = p1;
		}
	}

	void linePlotMain(Point pointStart, Point pointEnd, int b, COLORREF color = RGB(0, 0, 0)) {
		HWND hwnd = GetConsoleWindow();
		HDC hdc = GetDC(hwnd);
		HPEN hpen(CreatePen(PS_SOLID, 1, color));
		linePlot(hdc, hpen, Point((pointStart.X) / b + 100, (pointStart.Y) / b + 100), Point((pointEnd.X) / b + 100, (pointEnd.Y) / b + 100));

	}
	void linePlotMaina(Point pointStart, Point pointEnd, int b, COLORREF color = RGB(0, 255, 0)) {
		HWND hwnd = GetConsoleWindow();
		HDC hdc = GetDC(hwnd);
		HPEN hpen(CreatePen(PS_SOLID, 1, color));
		linePlot(hdc, hpen, Point((pointStart.X) / b, (pointStart.Y) / b), Point((pointEnd.X) / b, (pointEnd.Y) / b));

	}

	void polygonsPlot(Polygons& out_lines, COLORREF color = RGB(255, 0, 255)) {

		HWND hwnd = GetConsoleWindow();
		HDC hdc = GetDC(hwnd);
		int polygonsSize = out_lines.size();
		HPEN hpen(CreatePen(PS_SOLID, 1, color));
		for (int i = 0; i < polygonsSize; ++i) {
			polygonPlot(hdc, hpen, out_lines[i]);
		}
	}
}