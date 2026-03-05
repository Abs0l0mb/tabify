'use strict';

import {
    Table,
    TableRow, 
    Block, 
    Api,
    AddCategoryPopup,
    EditCategoryPopup,
    DeleteCategoryPopup,
    Button
} from '@src/classes';

export class TeamCategoriesTable extends Table {

    constructor(private teamId: number, parent: Block) {

        super({
            selectable: false,
            rowOptions: [
                {
                    text: 'Edit',
                    event: 'edit'
                },
                {
                    text: 'Delete',
                    event: 'delete'
                }
            ],
            actions: [
                {
                    text: 'Add category',
                    event: 'add'
                }
            ],
        }, parent);

        this.on('edit', (row: TableRow) => {
            new EditCategoryPopup(row.rowData.ID)
                .on('success', () => this.asyncPopulation());
        });

        this.on('delete', (row: TableRow) => {
            new DeleteCategoryPopup(row.rowData.ID)
                .on('done', row.delete.bind(row));
        });

        this.on('add', (button: Button) => {
            button.load();
            let popup = new AddCategoryPopup(this.teamId);
            popup.on('success', () => this.asyncPopulation());
            popup.on('hide', () => {
                button.unload();
            });
        });

        this.asyncPopulation();
    }
    
    /*
    **
    **
    */
    private async asyncPopulation() : Promise<void> {

        let rows: any[] = [];
        
        let entries: any[] = await Api.get('/team/categories', {
            teamId: this.teamId
        });

        let columns = {
            'ID': Table.NUMBER,
            'Title': Table.STRING,
            'Description': Table.STRING
        }

        for (let entry of entries) {
            
            rows.push({
                'ID': entry.id,
                'Title': entry.title,
                'Description': entry.description
            });
        }

        this.populate(columns, rows);
    }
}