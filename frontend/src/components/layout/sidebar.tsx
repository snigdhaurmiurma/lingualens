"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Search,
  Database,
  FlaskConical,
  Boxes,
  Scale,
  FileBarChart,
  Settings,
  Globe2,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  {
    name: "Overview",
    href: "/",
    icon: LayoutDashboard,
    badge: null,
  },
  {
    name: "Query Lab",
    href: "/query-lab",
    icon: Search,
    badge: "Interactive",
  },
  {
    name: "Datasets",
    href: "/datasets",
    icon: Database,
    badge: null,
  },
  {
    name: "Experiments",
    href: "/experiments",
    icon: FlaskConical,
    badge: null,
  },
  {
    name: "Model Registry",
    href: "/models",
    icon: Boxes,
    badge: "M2",
  },
  {
    name: "Fairness Analyzer",
    href: "/fairness",
    icon: Scale,
    badge: "M6 Preview",
  },
  {
    name: "Research Reports",
    href: "/reports",
    icon: FileBarChart,
    badge: "M7 Preview",
  },
  {
    name: "System & Hardware",
    href: "/settings",
    icon: Settings,
    badge: null,
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-border bg-card/60 backdrop-blur-md flex flex-col shrink-0 h-screen sticky top-0">
      {/* Brand Header */}
      <div className="h-16 flex items-center px-6 border-b border-border/80 gap-3">
        <div className="h-9 w-9 rounded-lg bg-primary/10 border border-primary/25 flex items-center justify-center text-primary shadow-sm">
          <Globe2 className="h-5 w-5" />
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <span className="font-semibold text-foreground tracking-tight text-base">
              LinguaLens
            </span>
            <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-primary/15 text-primary border border-primary/20">
              v0.1
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground leading-none mt-0.5">
            Fairness & IR Evaluation
          </p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
          Platform
        </div>

        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          const Icon = item.icon;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-150",
                isActive
                  ? "bg-primary text-primary-foreground shadow-sm shadow-primary/20 font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
              )}
            >
              <Icon
                className={cn(
                  "h-4 w-4 shrink-0 transition-colors",
                  isActive
                    ? "text-primary-foreground"
                    : "text-muted-foreground group-hover:text-foreground"
                )}
              />
              <span className="truncate">{item.name}</span>
              {item.badge && (
                <span
                  className={cn(
                    "ml-auto text-[10px] font-medium px-1.5 py-0.5 rounded",
                    isActive
                      ? "bg-primary-foreground/20 text-primary-foreground"
                      : "bg-secondary text-muted-foreground"
                  )}
                >
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Bottom Language Coverage Pill */}
      <div className="p-4 border-t border-border/80 bg-muted/20">
        <div className="text-[11px] font-medium text-muted-foreground mb-1.5">
          Active Language Matrix
        </div>
        <div className="flex flex-wrap gap-1">
          {["en", "bn", "bn-Latn", "hi", "ar"].map((lang) => (
            <span
              key={lang}
              className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-secondary/80 text-foreground border border-border"
            >
              {lang}
            </span>
          ))}
        </div>
      </div>
    </aside>
  );
}
