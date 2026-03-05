'use strict';

import { 
    TitledPage,
    VerticalSplit,
    TableRow,
    TeamsTable,
    TeamDetails
} from '@src/classes';

export class TeamsPage extends TitledPage {

    private split: VerticalSplit;

    constructor() {

        super('Teams', 'teams');
        
        this.split = new VerticalSplit(this.content);

        new TeamsTable(this.split.topContainer.empty())
            .addClass('light-zone')
            .on('select', (tableRow: TableRow) => {
                
                new TeamDetails(tableRow.rowData.ID, this.split.bottomContainer.empty());
            });
    }
}