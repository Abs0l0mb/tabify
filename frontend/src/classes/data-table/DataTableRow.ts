'use strict';

import { 
    Block, 
    Div,
    DataTable,
    DataTableCell,
    ContextMenu,
    ContextMenuOption 
} from '@src/classes';

export class DataTableRow extends Block {

    static readonly HEIGHT: number = 29;

    public cells: any = {};

    constructor(private rowData: any, private table: DataTable, private tbody: Block) {

        super('tr', 'data-row', tbody);

        const manifest = this.table.getManifest();
        const columns = this.table.getManifestColumns();

        if (!manifest)
            return;

        this.onNative('click', this.select.bind(this));
        this.onNative('contextmenu', (event: MouseEvent) => {
            event.preventDefault();
            this.displayContextMenu(event);
        });

        //============
        //OPTIONS CELL
        //============

        if (this.table.configuration.rowOptions) {

            let optionsCell = new Block('td', 'options-cell', this);

            new Div('options-icon', optionsCell).onNative('click', this.displayContextMenu.bind(this));
        }

        //==========
        //DATA CELLS
        //==========

        for (let key in this.rowData) {

            if (this.table.getManifestColumn(key)?.hidden)
                continue;

            const column = columns.get(key);

            this.cells[key] = new DataTableCell(this.rowData[key], column ? column.type : 'STRING', this);

            if (parseFloat(this.rowData[key]) === 0.0 
             || !this.rowData[key]
             || this.rowData[key] === 'null'
             || this.rowData[key] === '')
                this.cells[key].setData('priority', -1);
        }
    }

    /*
    **
    **
    */
    public select() : void {

        for (let row of this.table.getRows())
            row.setData('selected', row.uid === this.uid ? 1 : 0);
        
        for (let row of this.table.getRows())
            row.setData('options-displayed', 0);

        this.table.emit('select', this);
    }

    /*
    **
    **
    */
    public update(data: any) : void {

        for (let key in data) {

            let cell = this.cells[key];

            if (!cell)
                continue;
                
            cell.setValue(data[key]);

            this.rowData[key] = data[key];
        }
    }

    /*
    **
    **
    */
    private displayContextMenu(event: MouseEvent) : void {

        if (!this.table.configuration.rowOptions || this.table.configuration.rowOptions.length === 0)
            return;
            
        event.stopPropagation();
        
        if (!this.table.configuration.rowOptions)
            return;

        let options: ContextMenuOption[] = [];

        for (let option of this.table.configuration.rowOptions) {
            
            options.push({
                text: option.text,
                callback: () => {
                    this.table.emit(option.event, this);
                }
            });
        }

        this.setData('context-menu', 1);

        let contextMenu = new ContextMenu(event.pageX, event.pageY, options);
        
        contextMenu.on('hide', () => {
            this.setData('context-menu', 0);
        });
    }

    /*
    **
    **
    */
    public getRowData() : any {

        return this.rowData;
    }
}