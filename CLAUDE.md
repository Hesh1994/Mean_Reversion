# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the **Project-Initiation** repository for the [Future-Financial-Engineers](https://github.com/Future-Financial-Engineers) GitHub organization. It is in early scaffolding stage — currently contains a placeholder `Indicators` file and a minimal README.

## Intended Direction

Based on the repository name and organization focus (financial engineering), this project is expected to grow into a collection of financial indicator implementations, trading strategy scaffolding, or onboarding materials for the org.

## Current Files

- `README.md` — project title only; should be expanded as scope solidifies
- `Indicators` — empty placeholder; likely the future home of indicator definitions, specs, or code

## Development Notes

- Python environment: **Anaconda** (consistent with the broader home-directory projects)
- When adding indicator logic, follow the **manual numpy/pandas implementation** pattern used in related projects (no `ta` library dependency), with EMA seeded from SMA: `alpha = 2/(period+1)`, first EMA = SMA over seed bars
- For any Streamlit dashboards added here, reuse the `@st.cache_data` / `@st.cache_resource` caching pattern
