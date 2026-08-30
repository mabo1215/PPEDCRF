@echo off
setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------
REM docs/build.bat [doc-name-without-extension]
REM
REM Compiles docs/<doc-name>.tex to PDF directly from the source.
REM With NO argument, compiles every docs/*.tex found and prints a
REM pass/fail summary at the end (exit code is non-zero if any failed).
REM With an argument, compiles just that one doc, e.g.
REM   docs\build.bat experiment_progress
REM All intermediate LaTeX files (.aux/.log/.out/.pdf) are written under
REM docs/build/ via pdflatex -output-directory; the source .tex is never
REM copied, so docs/<doc-name>.tex is the single source of truth.
REM The final PDF is copied from docs/build/ back to docs/.
REM
REM Examples:
REM   docs\build.bat                       (compile every docs\*.tex)
REM   docs\build.bat experiment_progress   (compile just that one)
REM ---------------------------------------------------------------------

set "DOCS_DIR=%~dp0"
set "BUILD_DIR=%DOCS_DIR%build"

where pdflatex >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pdflatex not found on PATH. Install a LaTeX distribution ^(e.g. MiKTeX or TeX Live^) and ensure pdflatex is on PATH.
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

if not "%~1"=="" (
    call :build_one "%~1"
    exit /b !errorlevel!
)

REM No argument: compile every docs\*.tex.
set "OKCOUNT=0"
set "FAILCOUNT=0"
for %%F in ("%DOCS_DIR%*.tex") do (
    call :build_one "%%~nF"
    if !errorlevel! neq 0 (
        set /a FAILCOUNT+=1
    ) else (
        set /a OKCOUNT+=1
    )
)
echo.
echo [SUMMARY] !OKCOUNT! succeeded, !FAILCOUNT! failed.
if !FAILCOUNT! gtr 0 exit /b 1
exit /b 0

:build_one
set "DOC=%~1"
set "SOURCE=%DOCS_DIR%%DOC%.tex"

if not exist "%SOURCE%" (
    echo [ERROR] %SOURCE% not found.
    exit /b 1
)

pushd "%DOCS_DIR%"

pdflatex -interaction=nonstopmode -halt-on-error -aux-directory="%BUILD_DIR%" -output-directory="%BUILD_DIR%" "%SOURCE%" > "%BUILD_DIR%\%DOC%_pass1.log" 2>&1
if errorlevel 1 (
    echo [ERROR] pdflatex first pass failed for %DOC%.tex. See build\%DOC%_pass1.log
    popd
    exit /b 1
)

pdflatex -interaction=nonstopmode -halt-on-error -aux-directory="%BUILD_DIR%" -output-directory="%BUILD_DIR%" "%SOURCE%" > "%BUILD_DIR%\%DOC%_pass2.log" 2>&1
if errorlevel 1 (
    echo [ERROR] pdflatex second pass failed for %DOC%.tex. See build\%DOC%_pass2.log
    popd
    exit /b 1
)

popd

if not exist "%BUILD_DIR%\%DOC%.pdf" (
    echo [ERROR] %DOC%.pdf was not produced.
    exit /b 1
)

copy /Y "%BUILD_DIR%\%DOC%.pdf" "%DOCS_DIR%%DOC%.pdf" >nul
echo [OK] %DOC%.pdf compiled and copied to docs\%DOC%.pdf
exit /b 0
