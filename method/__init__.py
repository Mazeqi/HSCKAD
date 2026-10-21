# -*- coding: utf-8 -*-
"""HSCKAD method modules.

``qwen_vlm`` (stage 1) and ``sam3_promptable`` (stage 2) are imported lazily on
purpose: the two stages run in different environments (vLLM vs. SAM 3), so this
package must not pull in either dependency eagerly.
"""
