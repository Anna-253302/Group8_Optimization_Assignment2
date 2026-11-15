"""Plotting functions for Assignment 2."""

from pathlib import Path
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


DP_COLORS = {'cap_only':'#0072B2', 'BB':'#E69F00', 'soft_IR':'#CC79A7', 'BB_IR':'#009E73'}
DP_LABELS = {'cap_only':'C-supported DP', 'BB':'BB', 'soft_IR':'Soft IR', 'BB_IR':'BB+IR'}
REPORT_STYLE = {'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':9, 'axes.titlesize':10,
                'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
                'axes.spines.top':False, 'axes.spines.right':False,
                'pdf.fonttype':42, 'ps.fonttype':42}
# Figure 5's approved PDF width. All report PDFs use this width so inclusion at
# the same LaTeX width also gives the same printed label/tick/legend sizes.
REPORT_EXPORT_WIDTH_IN = 476.4 / 72


def _report_style(function):
    """Apply Figure 5's font roles independently of the caller's rc settings."""
    @wraps(function)
    def styled(*args,**kwargs):
        with plt.rc_context(REPORT_STYLE):
            return function(*args,**kwargs)
    return styled


def _label_panels(axes):
    """Add letters in reading order; descriptions belong in LaTeX captions."""
    for index,ax in enumerate(np.asarray(axes,dtype=object).ravel()):
        ax.text(0,1.025,f'({chr(ord("a")+index)})',transform=ax.transAxes,
                ha='left',va='bottom',fontsize=10,fontweight='bold')


def _dp_key(label):
    text = label.lower()
    if 'soft' in text:return 'soft_IR'
    if text in ('bb_ir','bb+ir'):return 'BB_IR'
    if text.startswith('bb'):return 'BB'
    return 'cap_only'


def _save_report_figure(fig, output_dir, stem):
    """Export with fixed physical width; fonts retain Figure 5's printed size."""
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    tight=fig.get_tightbbox(fig.canvas.get_renderer())
    if tight.width>REPORT_EXPORT_WIDTH_IN+1e-6:
        raise ValueError(f'{stem}: figure content exceeds the common export width')
    center=(tight.x0+tight.x1)/2
    bounds=Bbox.from_extents(center-REPORT_EXPORT_WIDTH_IN/2,tight.y0-.1,
                            center+REPORT_EXPORT_WIDTH_IN/2,tight.y1+.1)
    for ext in ('pdf','png'):
        fig.savefig(folder/f'{stem}.{ext}',dpi=220,bbox_inches=bounds,pad_inches=0)
    plt.close(fig)


def plot_dp_price_dispatch(cases, output_dir, *, stem='hourly_prices_imports'):
    """Aligned hourly prices [DKK/kWh] and imports [kW], from matched case JSONs."""
    with plt.rc_context(REPORT_STYLE):
        fig,axes=plt.subplots(2,1,figsize=(6.5,4.5),sharex=True,layout='constrained')
        first=next(iter(cases.values()))['hourly'];hours=np.array([r['hour'] for r in first])
        edges=np.r_[hours-.5,hours[-1]+.5]
        buy=np.array([r['discounted_import_price_DKK_per_kWh'] for r in first])
        sell=np.array([r['discounted_export_price_DKK_per_kWh'] for r in first])
        axes[0].stairs(buy,edges,color='#777777',linestyle='--',lw=.85,baseline=None)
        axes[0].stairs(sell,edges,color='#777777',linestyle=':',lw=.85,baseline=None)
        axes[0].fill_between(edges,np.r_[sell,sell[-1]],np.r_[buy,buy[-1]],step='post',
                             color='#777777',alpha=.18,label='External export-to-import band')
        for name,data in cases.items():
            rows=data['hourly'];key=_dp_key(name)
            if [r['hour'] for r in rows]!=hours.tolist():raise ValueError('Case hours differ')
            label=DP_LABELS[key]
            for ax,field in [(axes[0],'dp_internal_price_DKK_per_kWh'),(axes[1],'dp_import_kW')]:
                values=np.array([r[field]-(r['dp_export_kW'] if field=='dp_import_kW' else 0.) for r in rows],dtype=float)
                if not np.all(np.isfinite(values)):raise ValueError('Nonfinite plot values')
                ax.stairs(values,edges,color=DP_COLORS[key],lw=1.5,baseline=None,
                          linestyle={'cap_only':'-','BB':'--','soft_IR':'-.','BB_IR':'--'}[key],label=label)
        axes[1].stairs([r['cap_kW'] for r in first],edges,color='#555555',ls='--',lw=1,baseline=None,label='Contract cap')
        axes[0].set(ylabel='Posted price [DKK/kWh]')
        axes[1].set(ylabel='Net import [kW]',xlabel='Hour')
        _label_panels(axes)
        for ax in axes:
            ax.grid(axis='y',alpha=.17);ax.set_xlim(edges[0],edges[-1])
        axes[1].set_xticks([1,4,8,12,16,20,24])
        axes[0].legend(loc='upper left',ncol=2,fontsize=8)
        axes[1].legend(loc='lower left',ncol=2,fontsize=8)
        _save_report_figure(fig,output_dir,stem)


def plot_dp_contract_study(rows, output_dir):
    """Discrete cap cases, solver-bound intervals and all variants' import peaks.

    Horizontal offsets separate coincident outcomes. Filled markers are certified
    optima; open markers are incumbents. Error bars show solver bounds, not
    statistical confidence intervals. Unknown outcomes are shown in the status row.
    """
    from matplotlib.lines import Line2D
    factors=sorted({r['cap_factor'] for r in rows});x=np.arange(len(factors))
    with plt.rc_context(REPORT_STYLE):
        fig=plt.figure(figsize=(6.5,3.9),layout='constrained')
        grid=fig.add_gridspec(2,2,height_ratios=[1,.22])
        axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
        statuses=fig.add_subplot(grid[1,:]);statuses.axis('off')
        offset={'cap_only':-.13,'BB':0.,'BB_IR':.13}
        marker={'cap_only':'o','BB':'s','BB_IR':'D'}
        for variant in ['cap_only','BB','BB_IR']:
            for i,factor in enumerate(factors):
                r=next(v for v in rows if v['variant']==variant and v['cap_factor']==factor)
                if r['community_cost_DKK'] is None:continue
                color=DP_COLORS[variant];cert=bool(r['optimality_certified'])
                for ax,key in [(axes[0],'community_cost_DKK'),(axes[1],'peak_import_kW')]:
                    ax.plot(i+offset[variant],r[key],marker=marker[variant],ms=5,
                            mfc=color if cert else 'white',mec=color,mew=1.1,linestyle='none',zorder=4)
                bound=r.get('lower_bound_DKK')
                if not cert and bound is not None:
                    axes[0].errorbar(i+offset[variant],r['community_cost_DKK'],
                         yerr=[[max(0.,r['community_cost_DKK']-bound)],[0]],
                         color=color,capsize=3,lw=1,zorder=2)
        ref=next(r['R_cost_DKK'] for r in rows if r['R_cost_DKK'] is not None)
        axes[0].axhline(ref,color='#777777',ls='--',lw=.9)
        cap=[next(r['cap_kW'] for r in rows if r['cap_factor']==f) for f in factors]
        axes[1].plot(x,cap,color='#777777',ls='--',lw=.9)
        for ax in axes:
            ax.set_xticks(x,[f'{f:g}' for f in factors]);ax.set_xlabel('Contract case (cap factor f)')
            ax.grid(axis='y',alpha=.17);ax.set_xlim(-.45,len(factors)-.55)
        axes[0].set(ylabel='Community cost [DKK/day]')
        axes[1].set(ylabel='Community-meter import [kW]')
        _label_panels(axes)
        handles=[Line2D([],[],marker=marker[k],ls='',color=DP_COLORS[k],label=DP_LABELS[k]) for k in marker]
        handles += [Line2D([],[],color='#777777',ls='--',label='R cost / contract cap')]
        axes[1].legend(handles=handles,loc='upper left',fontsize=8)
        names={'PROVEN_INFEASIBLE':'Impossible','UNRESOLVED':'Unresolved',
               'OPTIMAL_CERTIFICATE':'Optimal','FEASIBLE_INCUMBENT':'Incumbent'}
        labels=[names[next(r['outcome'] for r in rows if r['cap_factor']==f and r['variant']=='BB_IR')] for f in factors]
        table=statuses.table(cellText=[['Hard IR vs R',*labels]],
                             colLabels=['Cap factor f',*[f'{f:g}' for f in factors]],loc='center',cellLoc='center')
        table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.12)
        for cell in table.get_celld().values():cell.set_linewidth(.35);cell.set_edgecolor('#aaaaaa')
        _save_report_figure(fig,output_dir,'cap_study')


def plot_dp_member_comparison(analyses, output_dir):
    """Daily gains against R, faceted by disutility class and ordered by asset type."""
    first=next(iter(analyses.values()))['members']
    ids={r['consumer_id'] for r in first};classes=sorted({r['slope_class'] for r in first})
    with plt.rc_context(REPORT_STYLE):
        fig,axes=plt.subplots((len(classes)+1)//2,2,figsize=(6.5,4.5),sharey=True,squeeze=False)
        for ax,slope in zip(axes.flat,classes):
            members=sorted([r for r in first if r['slope_class']==slope],key=lambda r:(r['has_pv'],r['has_battery']))
            # Load, PV, battery, PV+battery: the same order as the Q1 discussion.
            members=sorted(members,key=lambda r:2*int(r['has_battery'])+int(r['has_pv']))
            labels=[('PV+BAT' if r['has_battery'] and r['has_pv'] else 'BAT' if r['has_battery'] else 'PV' if r['has_pv'] else 'Load') for r in members]
            x=np.arange(len(members));width=.8/len(analyses)
            for j,(name,data) in enumerate(analyses.items()):
                own={r['consumer_id']:r for r in data['members']}
                if set(own)!=ids:raise ValueError('Selected runs must contain the same members')
                key=_dp_key(name)
                ax.bar(x-.4+(j+.5)*width,[own[m['consumer_id']]['gain_vs_R_DKK'] for m in members],
                       width=width,color=DP_COLORS[key],label=DP_LABELS[key])
            ax.axhline(0,color='#444444',lw=.9);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
            ax.set_xticks(x,labels)
            if ax in axes[:,0]:ax.set_ylabel('Gain vs R [DKK/day]')
        for ax in list(axes.flat)[len(classes):]:ax.set_visible(False)
        _label_panels(list(axes.flat)[:len(classes)])
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=len(analyses),frameon=False)
        fig.tight_layout(rect=(0,0,1,.92))
        _save_report_figure(fig,output_dir,'member_gains')


def plot_dp_response_ranges(ranges, hourly, output_dir):
    """Per-hour epsilon-optimal net-exchange envelope; not one daily schedule."""
    def total(key,h):return sum(float(r[key]) for r in ranges if int(r['hour'])==h)
    hours=np.array([int(r['hour']) for r in hourly]);edges=np.r_[hours-.5,hours[-1]+.5]
    lower=[total('minimum_net_import_kW',h) for h in hours]
    upper=[total('maximum_net_import_kW',h) for h in hours]
    with plt.rc_context(REPORT_STYLE):
        fig,ax=plt.subplots(figsize=(6.5,2.85),layout='constrained')
        ax.fill_between(edges,np.r_[lower,lower[-1]],np.r_[upper,upper[-1]],step='post',
                        color='#c7e2ef',label='Per-hour response range')
        ax.stairs([r['dp_import_kW']-r['dp_export_kW'] for r in hourly],edges,
                  color=DP_COLORS['cap_only'],lw=1.4,baseline=None,label='Selected C-supported net import')
        ax.stairs([r['cap_kW'] for r in hourly],edges,color='#555555',ls='--',lw=.9,baseline=None,label='Contract cap')
        ax.set(xlabel='Hour',ylabel='Net import [kW]',xlim=(edges[0],edges[-1]))
        ax.set_xticks([1,4,8,12,16,20,24])
        ax.legend(fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.02),borderaxespad=0)
        ax.grid(axis='y',alpha=.17)
        _save_report_figure(fig,output_dir,'response_ranges')


@_report_style
def plot_feeder_import(
    community,
    results,
    output_dir,
):
    """
    Compare hourly metered feeder imports against the DSO capacity limit.

    Parameters
    ----------
    community:
        CommunityData object containing the DSO capacity limit.

    results:
        Dictionary mapping model names to optimization results.
        Example: {"R": results_R, "I": results_I}

    output_dir:
        Directory where figures are saved.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capacity = np.asarray(
        community.contract.capacity_limit_kW,
        dtype=float,
    )

    hours = np.arange(1, len(capacity) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 3.25))

    for model_name, model_results in results.items():

        if model_name in {"C", "DP"}:

            # Community-meter import after internal netting
            feeder_import = np.asarray(
                model_results.community_import_kW,
                dtype=float,
            )

        else:

            # R and I: sum of individual gross imports
            feeder_import = np.sum(
                [
                    r.import_kW
                    for r in model_results.values()
                ],
                axis=0,
            )

        # A dashed DP curve keeps C visible when their selected imports coincide.
        style = dict(color="#9467bd", linestyle="--") if model_name == "DP" else {}
        ax.step(
            hours,
            feeder_import,
            where="mid",
            linewidth=2,
            label=f"Model {model_name}",
            **style,
        )
        
    # DSO capacity limit
    ax.step(
        hours,
        capacity,
        where="mid",
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="DSO capacity limit",
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Feeder import [kW]")

    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    _save_report_figure(fig,output_dir,'feeder_import')


@_report_style
def plot_settlement(result, output_dir):
    """Plot Q1 hourly prices (DKK/kWh) and daily member gains (DKK).

    Each benefit panel contains one slope class and four asset classes. Positive
    bars mean a member gains from C's dual settlement against the named model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hours = np.array([row["hour"] for row in result.hourly])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    series = [
        ("day_ahead_price_DKK_per_kWh", "Day-ahead", "#777777", ":"),
        ("discounted_import_price_DKK_per_kWh", "Discounted import", "#1f77b4", "--"),
        ("discounted_export_price_DKK_per_kWh", "Discounted export", "#ff7f0e", "--"),
        ("internal_price_DKK_per_kWh", "Internal settlement", "#2ca02c", "-"),
    ]
    for key, label, colour, style in series:
        ax.step(hours, [row[key] for row in result.hourly], where="mid",
                label=label, color=colour, linestyle=style, linewidth=1.8)
    first_binding = True
    for row in result.hourly:
        if row["import_kW"] >= row["cap_kW"] - result.summary["tolerance"]:
            ax.axvspan(row["hour"] - 0.5, row["hour"] + 0.5, color="#2ca02c", alpha=0.08,
                       label="Cap binding or exceeded" if first_binding else None)
            first_binding = False
    ax.set(xlabel="Hour of day", ylabel="Price [DKK/kWh]", xlim=(0.5, len(hours) + 0.5))
    ax.set_xticks(hours)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save_report_figure(fig,output_dir,'settlement_prices')

    _plot_member_gains(result.members, output_dir, stem="settlement_member_gains")


@_report_style
def _plot_member_gains(members, output_dir, *, stem, show_assets=False):
    """Daily gains in lettered panels, with disutility classes in sorted order."""
    classes = sorted({row["slope_class"] for row in members})
    ncols = min(2, len(classes))
    nrows = (len(classes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 1.9 * nrows), sharey=True, squeeze=False)
    for ax, slope in zip(axes.flat, classes):
        rows = [row for row in members if row["slope_class"] == slope]
        x = np.arange(len(rows))
        ax.bar(x - 0.18, [row["gain_vs_R_DKK"] for row in rows], width=0.36,
               label="Gain vs R", color="#1f77b4")
        ax.bar(x + 0.18, [row["gain_vs_I_DKK"] for row in rows], width=0.36,
               label="Gain vs I", color="#ff7f0e")
        ax.axhline(0, color="black", linewidth=0.8)
        labels = [row['consumer_id'] for row in rows]
        if show_assets:
            labels = [row['consumer_id'] + '\nLoad' + ('+PV' if row['has_pv'] else '')
                      + ('+BAT' if row['has_battery'] else '') for row in rows]
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Member gain [DKK/day]")
        ax.grid(axis="y", alpha=0.25)
    for ax in list(axes.flat)[len(classes):]:
        ax.set_visible(False)
    _label_panels(list(axes.flat)[:len(classes)])
    axes[0, 0].legend()
    fig.tight_layout()
    _save_report_figure(fig,output_dir,stem)


@_report_style
def plot_contract_sensitivity(rows, output_dir, dimension, base_value, member_count, thresholds=None):
    """Plot contract costs/exceedance and settlement outcomes against beta or f."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "Tariff discount beta" if dimension == "discount" else "Cap-energy factor f"
    colors = {"R": "#1f77b4", "I": "#ff7f0e", "C": "#2ca02c"}
    by_model = {kind: sorted([row for row in rows if row["model"] == kind],
                             key=lambda row: row["parameter_value"]) for kind in colors}
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
    for kind, records in by_model.items():
        x = [r["parameter_value"] for r in records]
        axes[0].plot(x, [r["total_objective_DKK"] for r in records], color=colors[kind],
                     label=kind, linewidth=2, linestyle="--" if kind == "R" else "-")
        axes[1].plot(x, [r["aggregate_metered_exceedance_kWh"] for r in records],
                     color=colors[kind], linewidth=2, linestyle="--" if kind == "R" else "-")
        axes[2].plot(x, [r["metered_peak_import_kW"] for r in records], label=kind,
                     color=colors[kind], linewidth=2, linestyle="--" if kind == "R" else "-")
    axes[0].set_ylabel("Community dispatch cost [DKK/day]")
    axes[1].set_ylabel("Aggregate metered exceedance\n[kWh/day]")
    axes[2].set_ylabel("Peak metered import [kW]")
    axes[2].set_xlabel(label)
    axes[2].plot([r["parameter_value"] for r in by_model["C"]],
                 [r["cap_max_kW"] for r in by_model["C"]], color="black", linestyle="-.",
                 label="Max hourly cap", linewidth=1)
    axes[2].legend(ncol=4, fontsize=8)
    _label_panels(axes)
    if thresholds:
        reference_cost = by_model["R"][0]["total_objective_DKK"]
        for kind in ("I", "C"):
            for bracket in thresholds[kind]["break_even_brackets"]:
                low, high = bracket["lower"], bracket["upper"]
                middle = (low + high) / 2
                axes[0].scatter([middle], [reference_cost], color=colors[kind], s=25, zorder=5)
                axes[0].annotate(f"{kind} break-even\n[{low:.3f}, {high:.3f}]",
                                 xy=(middle, reference_cost), xytext=(100, 50) if kind == "I" else (20, 20),
                                 textcoords="offset points", fontsize=8, color=colors[kind],
                                 bbox=dict(facecolor="white", edgecolor="none", alpha=.85),
                                 arrowprops=dict(arrowstyle="-", color=colors[kind]))
    for ax in axes:
        ax.axvline(base_value, color="#777777", linestyle=":", label="Base contract")
        ax.grid(alpha=.25)
    axes[0].legend(ncol=4)
    fig.tight_layout()
    _save_report_figure(fig,output_dir,'costs_and_exceedance')

    records = by_model["C"]
    x = [r["parameter_value"] for r in records]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)
    axes[0].plot(x, [r["manager_surplus_DKK"] for r in records], color=colors["C"], linewidth=2)
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_ylabel("Manager surplus [DKK/day]")
    for kind in ("R", "I"):
        axes[1].plot(x, [r[f"members_not_worse_off_vs_{kind}"] for r in records],
                     label=f"Compared with {kind}", color=colors[kind], linewidth=1,
                     marker="o", markersize=3, linestyle=":")
    axes[1].set(xlabel=label, ylabel="Members not worse off", ylim=(-.5, member_count + .5))
    axes[1].set_yticks(sorted(set([0, member_count // 4, member_count // 2,
                                 3 * member_count // 4, member_count])))
    axes[1].legend()
    for ax in axes:
        ax.axvline(base_value, color="#777777", linestyle=":")
        ax.grid(alpha=.25)
    _label_panels(axes)
    fig.tight_layout()
    _save_report_figure(fig,output_dir,'settlement_sensitivity')

    plot_report_contract_sensitivity(rows, output_dir, dimension, base_value, thresholds)


@_report_style
def plot_report_contract_sensitivity(rows, output_dir, dimension, base_value, thresholds=None):
    """Save concise Q1 figures: costs (DKK/day), and cap-study peaks (kW).

    The discount figure uses every point. The cap figure focuses on sampled
    factors 0.8--2.5, covering the base case and both default break-even brackets.
    If a custom sweep has no points in that window, its full range is shown.
    Full-range figures and all numerical results are retained by the caller.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"R": "#1f77b4", "I": "#ff7f0e", "C": "#2ca02c"}
    selected = rows
    if dimension == "cap":
        focused = [r for r in rows if 0.8 <= r["parameter_value"] <= 2.5]
        if focused:
            selected = focused
    by_model = {
        kind: sorted([r for r in selected if r["model"] == kind],
                     key=lambda r: r["parameter_value"])
        for kind in colors
    }
    x = [r["parameter_value"] for r in by_model["C"]]
    nrows = 1 if dimension == "discount" else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(6.5, 2.5 if nrows == 1 else 3.9),
                             sharex=True, squeeze=False)
    cost_ax = axes[0, 0]
    for kind, records in by_model.items():
        values = [r["parameter_value"] for r in records]
        style = "--" if kind == "R" else "-"
        cost_ax.plot(values, [r["total_objective_DKK"] for r in records],
                     color=colors[kind], linestyle=style, linewidth=2, label=f"Model {kind}")
        if nrows == 2:
            axes[1, 0].plot(values, [r["metered_peak_import_kW"] for r in records],
                            color=colors[kind], linestyle=style, linewidth=2)
    cost_ax.set_ylabel("Community cost [DKK/day]")
    if nrows == 2:
        _label_panels(axes)
    if thresholds:
        reference_cost = by_model["R"][0]["total_objective_DKK"]
        for kind in ("I", "C"):
            for bracket in thresholds[kind]["break_even_brackets"]:
                low, high = bracket["lower"], bracket["upper"]
                middle = (low + high) / 2
                if min(x) <= middle <= max(x):
                    cost_ax.scatter([middle], [reference_cost], color=colors[kind], s=25, zorder=5)
                    cost_ax.annotate(f"{kind} break-even: {middle:.3f}",
                                     xy=(middle, reference_cost), xytext=(12, 16),
                                     textcoords="offset points", color=colors[kind], fontsize=8,
                                     bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
    for ax in axes.flat:
        if min(x) <= base_value <= max(x):
            ax.axvline(base_value, color="#777777", linestyle=":", label="Base contract")
        ax.grid(alpha=.3)
    cost_ax.legend(ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Tariff discount beta" if dimension == "discount" else "Cap-energy factor f")
    bottom_margin = .02
    if nrows == 2:
        peak_ax = axes[1, 0]
        peak_ax.plot(x, [r["cap_max_kW"] for r in by_model["C"]],
                     color="black", linestyle="-.", linewidth=1.3, label="Hourly cap")
        peak_ax.set_ylabel("Peak metered import [kW]")
        peak_ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    _save_report_figure(fig,output_dir,'report_contract_sensitivity')
