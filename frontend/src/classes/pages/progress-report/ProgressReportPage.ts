'use strict';

import {
    TitledPage,
    ClientLocation,
    ProgressReportFilters,
    ProgressReport
} from '@src/classes';

declare var htmlDocx: any;

export class ProgressReportPage extends TitledPage {

    private filters: ProgressReportFilters;
    private report: ProgressReport;

    constructor() {

        super('Progress report', 'progress-report-page');
        
        this.content.addClass('light-zone');
        
        const canManageTeams = ClientLocation.get().api.accountData?.access_right_names?.includes('MANAGE TEAMS');
        const urlParam = ClientLocation.get().router.getParams().teamId;
        const myTeamId = ClientLocation.get().api.accountData.team_id;
        const teamId = canManageTeams && typeof urlParam === 'number' ? urlParam : myTeamId;

        this.filters = new ProgressReportFilters(this.content);

        this.filters.on('value', (data: any) => {
            this.render(teamId, data.startDate, data.endDate);
        });

        this.render(teamId, this.filters.getStartDate(), this.filters.getEndDate());
    }

    /*
    **
    **
    */
    private render(teamId: number, startDate: Date, endDate: Date) : void {

        if (this.report)
            this.report.delete();

        this.report = new ProgressReport(teamId, startDate, endDate, this.content);

        this.filters.on('generate', () => {

            const htmlContent = this.report.generateExportHTML();
            const converted = htmlDocx.asBlob(htmlContent);
            const blobUrl = URL.createObjectURL(converted);
            const downloadLink = document.createElement('a');
            downloadLink.href = blobUrl;
            downloadLink.download = 'GlobalProgressReport.docx';
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            URL.revokeObjectURL(blobUrl);
        })
    }
}