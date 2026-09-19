import { useMemo } from "react";
import katex from "katex";
export function MathText({
  latex,
  className = "",
}: {
  latex: string;
  className?: string;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(latex, {
        throwOnError: false,
        trust: false,
        output: "htmlAndMathml",
      }),
    [latex],
  );
  return (
    <span
      className={`math ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
