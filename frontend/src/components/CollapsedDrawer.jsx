import Card from './Card'
import CollapsibleSection from './CollapsibleSection'

// CollapsedDrawer — the UI-grammar unit for demoted sections: a slim
// header row that carries the glanceable signal as a live count badge,
// with the full content one tap away. Collapsed by default.
//
// Grammar (ui-revamp, Jul-30): a page answers its core questions with
// always-visible content; everything else becomes a drawer whose HEADER
// must still say something true and current ("5 upcoming", "last 6h ago",
// "Heat: 10%") — a bare title with a chevron is a missed signal. See the
// Command Center right column for the reference usage.
export default function CollapsedDrawer({ title, titleId, badge, defaultOpen = false, onOpenChange, cardProps = {}, children }) {
  return (
    <Card variant="glass" {...cardProps}>
      <CollapsibleSection
        title={title}
        titleId={titleId}
        badge={badge}
        defaultOpen={defaultOpen}
        onOpenChange={onOpenChange}
      >
        {children}
      </CollapsibleSection>
    </Card>
  )
}
