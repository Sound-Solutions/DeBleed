#include "ClickMonitor.h"

#if JUCE_DEBUG && JUCE_MAC
#import <Cocoa/Cocoa.h>
#import <objc/runtime.h>
#include <dlfcn.h>
#import <objc/message.h>


// ---- Debug-only hooks into ViewBridge's service-side marshal (private AppKit; never shipped). ----
// The marshal decides which host clicks reach our views; these log its geometry and decisions.
namespace
{
    std::function<void (const juce::String&)>* marshalLog = nullptr;

    typedef CGError (*GetRegionBoundsFn) (void*, CGRect*);
    typedef bool (*RegionIsEmptyFn) (void*);
    GetRegionBoundsFn getRegionBounds = nullptr;
    RegionIsEmptyFn regionIsEmpty = nullptr;

    juce::String rectStr (CGRect r)
    {
        return juce::String (r.origin.x, 1) + "," + juce::String (r.origin.y, 1) + " " + juce::String (r.size.width, 1) + "x" + juce::String (r.size.height, 1);
    }

    juce::String regionStr (void* region)
    {
        if (region == nullptr) return "null";
        juce::String out = "region@" + juce::String::toHexString ((juce::pointer_sized_int) region);
        if (regionIsEmpty != nullptr) out << (regionIsEmpty (region) ? " empty" : " nonempty");
        if (getRegionBounds != nullptr) { CGRect b {}; if (getRegionBounds (region, &b) == kCGErrorSuccess) out << " bounds=" << rectStr (b); }
        return out;
    }

    template <typename T> T readIvar (id obj, const char* name)
    {
        T value {};
        if (Ivar iv = class_getInstanceVariable ([obj class], name))
            memcpy (&value, (char*) (__bridge void*) obj + ivar_getOffset (iv), sizeof (T));
        return value;
    }

    juce::String marshalState (id m)
    {
        juce::String out;
        out << " resizingRegion=" << regionStr (readIvar<void*> (m, "_windowResizingRegion"))
            << " visibleRegion=" << regionStr (readIvar<void*> (m, "_visibleRegion"))
            << " remoteFrameScreen=" << rectStr (readIvar<CGRect> (m, "_remoteViewFrameInScreenCoords"))
            << " styleMask=" << juce::String ((juce::int64) readIvar<unsigned long long> (m, "_clientRequestedStyleMask"))
            << " declinedMask=" << juce::String ((juce::int64) readIvar<unsigned> (m, "_declinedEventMask"));
        const NSEdgeInsets in = readIvar<NSEdgeInsets> (m, "_alignmentRectInsets");
        out << " alignInsets=" << juce::String (in.top, 1) << "," << juce::String (in.left, 1) << "," << juce::String (in.bottom, 1) << "," << juce::String (in.right, 1);
        return out;
    }

    void log (const juce::String& s) { if (marshalLog != nullptr) (*marshalLog) (s); }

    IMP swap (Class cls, SEL sel, IMP replacement)
    {
        Method m = class_getInstanceMethod (cls, sel);
        if (m == nullptr) { log ("hook missing: " + juce::String (sel_getName (sel))); return nullptr; }
        return method_setImplementation (m, replacement);
    }

    BOOL (*origResizingContains) (id, SEL, CGPoint) = nullptr;
    BOOL hookResizingContains (id self, SEL sel, CGPoint p)
    {
        const BOOL r = origResizingContains (self, sel, p);
        log ("marshal eventResizingRegionContainsPoint " + juce::String (p.x, 1) + "," + juce::String (p.y, 1) + " -> " + juce::String ((int) r) + marshalState (self));
        return r;
    }

    void (*origSafeArea) (id, SEL, NSEdgeInsets) = nullptr;
    void hookSafeArea (id self, SEL sel, NSEdgeInsets in)
    {
        log ("marshal safeAreaInsetsDidChange " + juce::String (in.top, 1) + "," + juce::String (in.left, 1) + "," + juce::String (in.bottom, 1) + "," + juce::String (in.right, 1));
        origSafeArea (self, sel, in);
    }

    void (*origHostMouse) (id, SEL, unsigned long long) = nullptr;
    void hookHostMouse (id self, SEL sel, unsigned long long type)
    {
        log ("marshal hostWindowReceivedMouseEventType " + juce::String ((juce::int64) type) + marshalState (self));
        origHostMouse (self, sel, type);
    }

    void (*origRemoteLeft) (id, SEL, long long) = nullptr;
    void hookRemoteLeft (id self, SEL sel, long long v)
    {
        log ("marshal remoteViewReceivedLeftMouseDown " + juce::String ((juce::int64) v));
        origRemoteLeft (self, sel, v);
    }

    void (*origWindowLeft) (id, SEL, id) = nullptr;
    void hookWindowLeft (id self, SEL sel, id e)
    {
        log ("marshal windowReceivedLeftMouseDown " + juce::String ([[(NSObject*) e description] UTF8String]));
        origWindowLeft (self, sel, e);
    }

    void (*origOpaqueViews) (id, SEL, id, void*, CGRect) = nullptr;
    void hookOpaqueViews (id self, SEL sel, id window, void* region, CGRect rect)
    {
        log ("marshal window:hasRegionForOpaqueViews " + regionStr (region) + " blockingDraggableFrame " + rectStr (rect));
        origOpaqueViews (self, sel, window, region, rect);
    }

    BOOL (*origSendEventTo) (id, SEL, id, id) = nullptr;
    BOOL hookSendEventTo (id self, SEL sel, id e, id to)
    {
        const BOOL r = origSendEventTo (self, sel, e, to);
        if ([(NSEvent*) e type] == NSEventTypeLeftMouseDown)
            log ("marshal sendEvent:to: " + juce::String (class_getName ([to class])) + " -> " + juce::String ((int) r) + " " + juce::String ([[(NSObject*) e description] UTF8String]));
        return r;
    }

    BOOL (*origDragWindow) (id, SEL, id, CGPoint) = nullptr;
    BOOL hookDragWindow (id self, SEL sel, id w, CGPoint p)
    {
        const BOOL r = origDragWindow (self, sel, w, p);
        log ("marshal dragWindow:relativeToMouseDown: " + juce::String (p.x, 1) + "," + juce::String (p.y, 1) + " -> " + juce::String ((int) r));
        return r;
    }


    // ---- The jail window's drag region: where the host treats our window as draggable background. ----
    typedef CGRect (*RectGetter) (id, SEL);
    typedef void* (*RegionForRect) (id, SEL, CGRect);
    typedef void* (*RegionForDescendants) (id, SEL, CGRect, BOOL, BOOL);
    typedef BOOL (*BoolGetter) (id, SEL);
    typedef id (*IdGetter) (id, SEL);

    juce::String dragRegionReport (NSWindow* w)
    {
        juce::String out = "dragregion:";
        SEL sDraggable = sel_registerName ("_draggableFrame");
        SEL sIsDraggable = sel_registerName ("_isDraggable");
        SEL sBottomBar = sel_registerName ("_movableByBottomBar");
        SEL sDesc = sel_registerName ("_lastDragRegionDataDescription");
        SEL sRegionFor = sel_registerName ("_regionForOpaqueViewsBlockingDraggableFrame:");
        SEL sDescendants = sel_registerName ("_regionForOpaqueDescendants:forMove:forUnderTitlebar:");
        NSView* frameView = [[w contentView] superview];
        NSView* juceView = nil;
        for (NSView* v = [w contentView]; v != nil && [[v subviews] count] > 0; v = [[v subviews] firstObject]) juceView = [[v subviews] firstObject];
        CGRect draggable = CGRectZero;
        if ([w respondsToSelector: sDraggable]) draggable = ((RectGetter) objc_msgSend) (w, sDraggable);
        out << " frame=" << rectStr ([w frame]) << " draggableFrame=" << rectStr (draggable)
            << " isDraggable=" << ([w respondsToSelector: sIsDraggable] ? juce::String ((int) ((BoolGetter) objc_msgSend) (w, sIsDraggable)) : "?")
            << " movableByBottomBar=" << ([w respondsToSelector: sBottomBar] ? juce::String ((int) ((BoolGetter) objc_msgSend) (w, sBottomBar)) : "?")
            << " isMovable=" << (int) [w isMovable] << " movableBg=" << (int) [w isMovableByWindowBackground]
            << " styleMask=" << juce::String ((juce::int64) [w styleMask]) << " textured=" << (int) (([w styleMask] & 256) != 0)
            << " windowOpaque=" << (int) [w isOpaque] << " contentOpaque=" << (int) [[w contentView] isOpaque]
            << " frameViewOpaque=" << (int) [frameView isOpaque];
        if (juceView != nil)
            out << " deepest=" << class_getName ([juceView class]) << " opaque=" << (int) [juceView isOpaque]
                << " canMoveWindow=" << (int) [juceView mouseDownCanMoveWindow]
                << " layerOpaque=" << (int) ([juceView layer] != nil ? [[juceView layer] isOpaque] : -1);
        for (NSView* v = juceView; v != nil; v = [v superview])
            out << " | " << class_getName ([v class]) << " opaque=" << (int) [v isOpaque] << " moves=" << (int) [v mouseDownCanMoveWindow];
        if ([w respondsToSelector: sDesc])
        {
            id desc = ((IdGetter) objc_msgSend) (w, sDesc);
            out << "\n    lastDragRegion=" << (desc != nil ? juce::String ([[desc description] UTF8String]) : juce::String ("nil"));
        }
        if ([w respondsToSelector: sRegionFor])
        {
            const CGRect whole = CGRectMake (0, 0, [w frame].size.width, [w frame].size.height);
            void* r1 = ((RegionForRect) objc_msgSend) (w, sRegionFor, whole);
            out << "\n    opaqueViewsBlocking(whole " << rectStr (whole) << ")=" << regionStr (r1);
            void* r2 = ((RegionForRect) objc_msgSend) (w, sRegionFor, draggable);
            out << "  opaqueViewsBlocking(draggable)=" << regionStr (r2);
        }
        if (frameView != nil && [frameView respondsToSelector: sDescendants])
        {
            void* r3 = ((RegionForDescendants) objc_msgSend) (frameView, sDescendants, [frameView bounds], YES, NO);
            out << "\n    frameView opaqueDescendants(forMove)=" << regionStr (r3);
        }
        return out;
    }

    void (*origSetLastDragRegion) (id, SEL, void*) = nullptr;
    void hookSetLastDragRegion (id self, SEL sel, void* region)
    {
        log ("window _setLastDragRegion: " + regionStr (region) + " frame=" + rectStr ([(NSWindow*) self frame]));
        origSetLastDragRegion (self, sel, region);
    }

    void installMarshalHooks()
    {
        static bool done = false;
        if (done) return;
        done = true;
        origSetLastDragRegion = (decltype (origSetLastDragRegion)) swap ([NSWindow class], sel_registerName ("_setLastDragRegion:"), (IMP) hookSetLastDragRegion);
        getRegionBounds = (GetRegionBoundsFn) dlsym (RTLD_DEFAULT, "CGSGetRegionBounds");
        regionIsEmpty = (RegionIsEmptyFn) dlsym (RTLD_DEFAULT, "CGSRegionIsEmpty");
        Class cls = objc_getClass ("NSViewServiceMarshal");
        if (cls == nil) { log ("hook: no NSViewServiceMarshal"); return; }
        origResizingContains = (decltype (origResizingContains)) swap (cls, sel_registerName ("eventResizingRegionContainsPoint:"), (IMP) hookResizingContains);
        origSafeArea = (decltype (origSafeArea)) swap (cls, sel_registerName ("remoteViewSafeAreaInsetsDidChange:"), (IMP) hookSafeArea);
        origHostMouse = (decltype (origHostMouse)) swap (cls, sel_registerName ("hostWindowReceivedMouseEventType:"), (IMP) hookHostMouse);
        origRemoteLeft = (decltype (origRemoteLeft)) swap (cls, sel_registerName ("remoteViewReceivedLeftMouseDown:"), (IMP) hookRemoteLeft);
        origWindowLeft = (decltype (origWindowLeft)) swap (cls, sel_registerName ("windowReceivedLeftMouseDown:"), (IMP) hookWindowLeft);
        origOpaqueViews = (decltype (origOpaqueViews)) swap (cls, sel_registerName ("window:hasRegionForOpaqueViews:blockingDraggableFrame:"), (IMP) hookOpaqueViews);
        origSendEventTo = (decltype (origSendEventTo)) swap (cls, sel_registerName ("sendEvent:to:"), (IMP) hookSendEventTo);
        origDragWindow = (decltype (origDragWindow)) swap (cls, sel_registerName ("dragWindow:relativeToMouseDown:"), (IMP) hookDragWindow);
        log ("marshal hooks installed; CGSGetRegionBounds=" + juce::String (getRegionBounds != nullptr ? 1 : 0) + " CGSRegionIsEmpty=" + juce::String (regionIsEmpty != nullptr ? 1 : 0));
    }
}

void installClickMonitor (std::function<void (const juce::String&)> log)
{
    static id monitor = nil;
    if (monitor != nil)
        return;

    auto shared = std::make_shared<std::function<void (const juce::String&)>> (std::move (log));
    marshalLog = shared.get();
    installMarshalHooks();
    monitor = [NSEvent addLocalMonitorForEventsMatchingMask: NSEventMaskLeftMouseDown
                                                     handler: ^NSEvent* (NSEvent* e)
    {
        NSWindow* w = [e window];
        const NSPoint p = [e locationInWindow];
        juce::String line = "NSEvent mouseDown";
        if (w == nil)
        {
            (*shared) (line + " (no window)");
            return e;
        }
        const NSRect f = [w frame];
        const NSPoint g = [NSEvent mouseLocation];
        line << " global=" << juce::String (g.x, 1) << "," << juce::String (g.y, 1)
             << " winorigin=" << juce::String (f.origin.x, 0) << "," << juce::String (f.origin.y, 0)
             << " screenH=" << juce::String ([[NSScreen mainScreen] frame].size.height, 0)
             << " evnum=" << juce::String ((long long) [e eventNumber])
             << " clicks=" << juce::String ((long long) [e clickCount]);
        line << " window '" << juce::String ([[w title] UTF8String] != nullptr ? [[w title] UTF8String] : "")
             << "' " << juce::String (f.size.width, 0) << "x" << juce::String (f.size.height, 0)
             << " style=" << juce::String ((long long) [w styleMask])
             << " loc=" << juce::String (p.x, 1) << "," << juce::String (p.y, 1) << " (from bottom-left)";
        NSView* frameView = [[w contentView] superview];
        auto describeChain = [] (NSView* hit, juce::String& out)
        {
            for (NSView* v = hit; v != nil; v = [v superview])
            {
                const NSRect r = [v frame];
                out << " > " << juce::String ([NSStringFromClass ([v class]) UTF8String])
                    << "[" << juce::String (r.origin.x, 0) << "," << juce::String (r.origin.y, 0)
                    << " " << juce::String (r.size.width, 0) << "x" << juce::String (r.size.height, 0) << "]";
            }
        };
        NSView* hit = [frameView hitTest: p];
        line << "  hit:";
        describeChain (hit, line);
        // Failing clicks carry no location; hit-test the real point (global minus window origin) instead.
        if (p.x < 0 && p.y < 0)
        {
            const NSPoint real = NSMakePoint (g.x - f.origin.x, g.y - f.origin.y);
            line << "  real=" << juce::String (real.x, 1) << "," << juce::String (real.y, 1) << " hit:";
            describeChain ([frameView hitTest: real], line);
            line << "  hidden=" << (int) [[w contentView] isHiddenOrHasHiddenAncestor];
        }
        // Raw event details: AppKit's own description plus the CGEvent's location and target windows.
        line << "\n    desc=" << juce::String ([[(NSObject*) e description] UTF8String]);
        if (CGEventRef ce = [e CGEvent])
        {
            const CGPoint cl = CGEventGetLocation (ce);
            const CGPoint ul = CGEventGetUnflippedLocation (ce);
            line << "\n    cg loc=" << juce::String (cl.x, 1) << "," << juce::String (cl.y, 1)
                 << " unflipped=" << juce::String (ul.x, 1) << "," << juce::String (ul.y, 1)
                 << " winUnder=" << juce::String ((long long) CGEventGetIntegerValueField (ce, (CGEventField) 91))
                 << " winCanHandle=" << juce::String ((long long) CGEventGetIntegerValueField (ce, (CGEventField) 92))
                 << " winNum=" << juce::String ((long long) [e windowNumber])
                 << " subtype=" << juce::String ((long long) [e subtype]);
        }
        else
            line << "\n    no CGEvent";
        line << "\n    " << dragRegionReport (w);
        (*shared) (line);
        // Once: the window/delegate classes and any method whose name smells like hit testing.
        static bool introspected = false;
        if (! introspected)
        {
            introspected = true;
            juce::String info = "introspect: window class " + juce::String (class_getName ([w class]))
                              + " delegate " + juce::String ([w delegate] != nil ? class_getName ([[w delegate] class]) : "nil")
                              + " ignoresMouse=" + juce::String ((int) [w ignoresMouseEvents])
                              + " opaque=" + juce::String ((int) [w isOpaque])
                              + " movableBg=" + juce::String ((int) [w isMovableByWindowBackground]);
            auto dumpMethods = [&info] (Class cls, const char* label)
            {
                for (Class c = cls; c != nil && c != [NSObject class]; c = class_getSuperclass (c))
                {
                    unsigned count = 0;
                    Method* methods = class_copyMethodList (c, &count);
                    for (unsigned i = 0; i < count; ++i)
                    {
                        juce::String name (sel_getName (method_getName (methods[i])));
                        auto lower = name.toLowerCase();
                        if (lower.contains ("hit") || lower.contains ("region") || lower.contains ("mousedown")
                            || lower.contains ("opaque") || lower.contains ("sendevent") || lower.contains ("fake"))
                            info << "\n    " << label << " " << class_getName (c) << " " << name;
                    }
                    free (methods);
                }
            };
            dumpMethods ([w class], "win");
            if ([w delegate] != nil) dumpMethods ([[w delegate] class], "del");
            dumpMethods ([[w contentView] class], "content");
            dumpMethods ([[[w contentView] superview] class], "frame");
            (*shared) (info);
        }
        // Once per event: the whole native view tree of the jail window, so a stray view shows up.
        static int dumps = 0;
        if (dumps++ < 3)
        {
            juce::String tree = "view tree:";
            std::function<void (NSView*, int)> walk = [&] (NSView* v, int depth)
            {
                const NSRect r = [v frame];
                tree << "\n" << juce::String::repeatedString ("  ", depth)
                     << juce::String ([NSStringFromClass ([v class]) UTF8String])
                     << " [" << juce::String (r.origin.x, 0) << "," << juce::String (r.origin.y, 0)
                     << " " << juce::String (r.size.width, 0) << "x" << juce::String (r.size.height, 0) << "]"
                     << ([v isHidden] ? " hidden" : "") << ([v wantsLayer] ? " layer" : "");
                for (NSView* c in [v subviews]) walk (c, depth + 1);
            };
            walk (frameView, 0);
            tree << "\nwindows in process: " << (int) [[NSApp windows] count];
            for (NSWindow* ow in [NSApp windows])
            {
                const NSRect r = [ow frame];
                tree << "\n  " << juce::String ([NSStringFromClass ([ow class]) UTF8String])
                     << " '" << juce::String ([[ow title] UTF8String]) << "' [" << juce::String (r.origin.x, 0) << "," << juce::String (r.origin.y, 0)
                     << " " << juce::String (r.size.width, 0) << "x" << juce::String (r.size.height, 0) << "]"
                     << " visible=" << (int) [ow isVisible] << " level=" << juce::String ((long long) [ow level]);
            }
            (*shared) (tree);
        }
        return e;
    }];
}
#else
void installClickMonitor (std::function<void (const juce::String&)>) {}
#endif
